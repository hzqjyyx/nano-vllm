from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # 所有Block对象的池子，通过block_id索引访问
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # prefix caching索引：通过block内容的hash快速找到可复用的block_id
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲block队列，按FIFO顺序分配（用deque支持高效的pop/append）
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 已使用block集合，用于判断cache hit的block是否需要增加引用计数（用set支持O(1)查找）
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        # 前置条件：sequence必须是新的，还没有分配过block
        assert not seq.block_table
        # h是累积的hash，用于链式计算：hash(block_i) = hash(hash(block_{i-1}), tokens_i)
        h = -1
        # cache_miss是"粘性"的：一旦miss，后续所有block都miss（因为prefix不再匹配）
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # 只有满block才能参与prefix caching，最后一个partial block的h保持-1
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            # 双重检查：hash碰撞保护，必须验证实际token_ids相同
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # cache miss路径：分配新block
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # cache hit路径：复用已有block
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # block已被其他sequence使用，增加引用计数（共享）
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # block在free状态但hash匹配（之前被deallocate但内容未清除）
                    block = self._allocate_block(block_id)
            # 更新block的hash和内容（cache miss时是新内容，cache hit时是重复更新）
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        # 从后往前遍历：后面的block更可能是独占的（ref_count=1），先释放可以更早回收内存
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            # 减少引用计数，可能有多个sequence共享同一个block（prefix caching）
            block.ref_count -= 1
            # 只有当没有任何sequence引用时才真正释放block
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 清理sequence的缓存状态，为可能的重新allocate做准备
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # decode阶段每次生成一个token后调用，维护block_table和prefix caching索引
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 刚填满上一个block，需要分配新block来容纳新生成的token
            assert last_block.hash != -1  # 上一个block应该已经被finalize（见下面的elif分支）
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 刚好填满当前block，finalize它：计算hash并加入prefix caching索引
            assert last_block.hash == -1  # 未满的block不应该有hash
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 当前block未满，继续填充，无需操作
            assert last_block.hash == -1
