
def decay(current, stamp,name='log', weight=0.1, coefficient=0.5):
    # smaller the better
    # log
    if name == 'log':
        pass
    elif name == 'poly':
        # x ^^ y
        return ((current - stamp) ** coefficient) * (weight)