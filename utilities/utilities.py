def chunker(array, chunk_size):

    return (array[pos: pos+chunk_size] for pos in xrange(0, len(array), chunk_size))
