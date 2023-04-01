class Progress_bar:
    def __init__(self, total_num:int, bar_size:int = 40) -> None:
        self._bar_format = '\rProgress: [{}{}]\t{:.2%}\tProcess: {}/{}'
        self._bar_size = bar_size
        self._total_num = total_num
        self._now_num = 0
        self._now_bar = 0
        self._now_white = 0
    def update(self, step:int = 1) -> None:
        self._now_num += step
        self._now_num %= self._total_num + 1

        self._now_bar = int((self._now_num / self._total_num) * self._bar_size)
        self._now_white = self._bar_size - self._now_bar

        print(self._bar_format.format(
            '='*self._now_bar,
            ' '*self._now_white,
            self._now_num / self._total_num,
            self._now_num,
            self._total_num
        ), end='')
        
        if self._now_num == self._total_num:
            print()