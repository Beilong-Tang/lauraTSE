import torch.distributed

HINTED = dict()

def hint_once(content, uid, rank=None, times = 1):
    """
    ranks: which rank to output log
    times: how many times to hint
    """
    _cur_rank = None 
    if torch.distributed.is_initialized():
        _cur_rank = torch.distributed.get_rank()

    if HINTED.get(uid) is None:
        HINTED[uid] = 0

    if (rank is None) or (_cur_rank is None) or _cur_rank == rank:

        # Check if the times exceed the HINTED
        if HINTED.get(uid) >=times:
            return 
        
        if _cur_rank is not None:
            print(f"[HINT_ONCE {HINTED.get(uid)+1/times}] Rank {_cur_rank}: {content}")
        else:
            print(f"[HINT_ONCE {HINTED.get(uid)+1/times}] {content}")
        HINTED[uid] = HINTED[uid] + 1

def check_hint(uid) -> bool:
    """
    Check if uid is in already hinted
    """
    return uid in HINTED

