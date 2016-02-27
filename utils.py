import progressbar as pb


def get_pbar(num, prefix=""):
    assert isinstance(prefix, str)
    pbar = pb.ProgressBar(widgets=[prefix, pb.Percentage(), pb.Bar(), pb.ETA()], maxval=num)
    return pbar
