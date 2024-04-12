from testbook import testbook


@testbook('../StartHere.ipynb', execute=True)
def test_func(tb):
    freq0 = tb.get("freq0")

    assert freq0 == 399723277333333.3