import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environment.exchange import Exchange


def test_basic_match():
    ex = Exchange()
    ex.place_quote('bid', 99.0, 100)
    ex.place_quote('ask', 101.0, 100)

    trades = ex.match_orders([{'side': 'buy', 'size': 10}])
    assert len(trades) == 1
    assert trades[0]['price'] == 101.0
    assert trades[0]['size'] == 10
    print("PASS: buy hits ask at 101.0")

    trades = ex.match_orders([{'side': 'sell', 'size': 5}])
    assert len(trades) == 1
    assert trades[0]['price'] == 99.0
    assert trades[0]['size'] == 5
    print("PASS: sell hits bid at 99.0")


def test_no_quote_no_trade():
    ex = Exchange()
    trades = ex.match_orders([{'side': 'buy', 'size': 10}])
    assert trades == []
    print("PASS: no quotes → no trades")


def test_partial_fill():
    ex = Exchange()
    ex.place_quote('ask', 101.0, 3)
    trades = ex.match_orders([{'side': 'buy', 'size': 10}])
    assert trades[0]['size'] == 3    # only 3 available
    assert ex.asks == {}             # quote fully consumed
    print("PASS: partial fill consumes quote")


def test_cancel_clears_book():
    ex = Exchange()
    ex.place_quote('bid', 99.0, 100)
    ex.cancel_all_quotes()
    trades = ex.match_orders([{'side': 'sell', 'size': 5}])
    assert trades == []
    print("PASS: cancel_all_quotes clears book")


if __name__ == '__main__':
    test_basic_match()
    test_no_quote_no_trade()
    test_partial_fill()
    test_cancel_clears_book()
    print("\nAll exchange tests passed.")
