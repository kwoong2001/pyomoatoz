
t = 7

intervals = [1, 7, 13, 19]
t_int = int(t)
t_start = max([v for v in intervals if v <= t_int], default=None)


print(f"t: {t}, t_start: {t_start}")