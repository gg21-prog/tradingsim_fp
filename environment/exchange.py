class Exchange:
    """
    Minimal order book. Market maker posts one bid and one ask.
    Incoming market orders match against the best available quote.
    """

    def __init__(self):
        self.bids = {}   # price -> size
        self.asks = {}   # price -> size

    def reset(self):
        self.bids = {}
        self.asks = {}

    def place_quote(self, side, price, size):
        if side == 'bid':
            self.bids[price] = size
        elif side == 'ask':
            self.asks[price] = size

    def cancel_all_quotes(self):
        self.bids = {}
        self.asks = {}

    def match_orders(self, orders):
        """
        Match a list of market orders against standing quotes.
        Each order: {'side': 'buy'|'sell', 'size': int}
        Returns list of trades: {'side': str, 'price': float, 'size': int}
        """
        trades = []
        for order in orders:
            if order['side'] == 'buy' and self.asks:
                best_ask = min(self.asks)
                fill = min(order['size'], self.asks[best_ask])
                trades.append({'side': 'buy', 'price': best_ask, 'size': fill})
                self.asks[best_ask] -= fill
                if self.asks[best_ask] == 0:
                    del self.asks[best_ask]

            elif order['side'] == 'sell' and self.bids:
                best_bid = max(self.bids)
                fill = min(order['size'], self.bids[best_bid])
                trades.append({'side': 'sell', 'price': best_bid, 'size': fill})
                self.bids[best_bid] -= fill
                if self.bids[best_bid] == 0:
                    del self.bids[best_bid]

        return trades
