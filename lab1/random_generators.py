x = 123456789
y = 987654321
z = 43219876
c = 6543217


def to_int32(number):
    return number % (2 ** 32)


def JKISS(x, y, z, c):
    x = to_int32(to_int32(314527869 * x) + 1234567)
    y ^= to_int32(y << 5)
    y ^= y >> 7
    y ^= to_int32(y << 22)
    t = ((4294584393 * z) % (2 ** 64) + c) % (2 ** 64)
    c = t >> 32
    z = to_int32(t)
    return to_int32(x + y + z), x, y, z, c


for i in range(0, 10):
    result, x, y, z, c = JKISS(x, y, z, c)
    # print(f"x={x}, y={y}, z={z}, c={c}")
    print(result / 4294967295)
