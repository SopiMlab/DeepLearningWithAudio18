# A small function that calculates how many times we can convolve with stride 2 until we get under the kernel size.
def count_convolutions(input_shape, kernel_size):
    count = 0
    x = input_shape[0]
    while x > kernel_size:
        x = x/2
        count += 1
    return count - 2