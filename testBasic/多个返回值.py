def fun(a):
    if a == 1:
        return False
    elif a == 2:
        return False,a
    else:
        return True,a

if __name__ == '__main__':
    [bb,cc] = fun(1)
    print(bb)