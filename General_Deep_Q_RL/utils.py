def appendCircular(array, newObj):
    array[0:-1] = array[1:]
    array[-1] = newObj
