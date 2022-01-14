import numba


@numba.jit
def get_color(img, background, colors):
    n = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c = img[i, j]
            if c[0] == background[0] and c[1] == background[1] and c[2] == background[2]: continue
            for k in range(colors.shape[0]):
                if colors[k, 0] == c[0] and colors[k, 1] == c[1] and colors[k, 2] == c[2]:
                    colors[k, 3] += 1
                    break
            else:
                colors[n, :3] = c
                colors[n, 3] = 1
                n += 1
                if n > colors.shape[0]: return
    return colors[list(colors[:, 3]).index(max(colors[:, 3])), :3]


@numba.jit
def check_match(img, m,
                color) -> bool:
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if not m[i, j, 3]: continue
            if m[i, j, 0]:
                if img[i, j, 0] != color[0] or img[i, j, 1] != color[1] or img[i, j, 2] != color[2]:
                    return False
            elif not m[i, j, 1]:
                if img[i, j, 0] == color[0] and img[i, j, 1] == color[1] and img[i, j, 2] == color[2]:
                    return False
    return True


@numba.jit
def round(x: float) -> int:
    if x >= 0.:
        return int(x) + ((x - int(x)) > .5)
    else:
        return int(x) - ((int(x) - x) > .5)


@numba.jit
def get_center(img, dot, color,
               r: int) -> tuple:
    x = y = 0
    n = 0
    for i in range(max(0, min(img.shape[0] - 1, dot[0] - r)), max(0, min(img.shape[0] - 1, dot[0] + r + 1))):
        for j in range(max(0, min(img.shape[1] - 1, dot[1] - r)), max(0, min(img.shape[1] - 1, dot[1] + r + 1))):
            if (dot[0] - i) ** 2 + (dot[1] - j) ** 2 <= r * r and img[i, j, 0] == color[0] and img[i, j, 1] == color[
                1] and img[i, j, 2] == color[2]:
                x += i
                y += j
                n += 1
    return round(x / n), round(y / n)


@numba.jit
def find_(img, m, center, color) -> list:
    matches = []
    for i in range(img.shape[0] - m.shape[0]):
        for j in range(img.shape[1] - m.shape[1]):
            if check_match(img[i: i + img.shape[0], j: j + img.shape[1]], m, color):
                matches.append((i + center[0], j + center[1]))
    return matches
