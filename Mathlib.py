pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
e = 0
factorials = {0: 1, 1: 1}
def fibonacci(n, printb=0):
    x = 0
    y = 1
    if printb == 1:
        print("1: " + str(y))
    for i in range(n - 1):
        z = x + y
        if printb == 1:
            print(str(i + 2) + ": " + str(z))
        x = y
        y = z
    if printb == 1:
        print("\n \n")
    return z
def sqrt(n):
    x = n
    count = 0

    while True:
        count += 1
        root = 0.5 * (x + (n / x))
        if (abs(root - x) < 0.00000000000000000000001):
            break
        x = root
    
    return root
def pyth_thereom(a, b):
    return sqrt((a * a) + (b * b))
def factorial(n):
    if n in factorials:
        return factorials[n]
    else:
        result = n * factorial(n - 1)
        factorials[n] = result
        return result
def sin(n, terms=170, decimal_places=12):
    angle_in_radians = n * (pi / 180)
    sin_value = angle_in_radians
    sign = -1
    for i in range(3, terms + 1, 2):
        sin_value += sign * (angle_in_radians ** i) / factorial(i)
        sign *= -1

    # Round the result to the specified number of decimal places
    rounded_result = round(sin_value, decimal_places)

    return rounded_result
def cos(n, terms=170, decimal_places=12):
    # Convert angle to radians
    angle_in_radians = n * (pi / 180.0)

    # Calculate cos using Taylor series expansion
    cos_value = 1
    sign = -1
    for i in range(2, terms + 1, 2):
        cos_value += sign * (angle_in_radians ** i) / factorial(i)
        sign *= -1

    # Round the result to the specified number of decimal places
    rounded_result = round(cos_value, decimal_places)

    return rounded_result
def tan(n, terms=170, decimal_places=12):
    # Calculate tan as sin(angle) / cos(angle)
    sin_value = sin(n, terms, decimal_places)
    cos_value = cos(n, terms, decimal_places)

    # Check for division by zero (cos(angle) = 0)
    if cos_value == 0:
        return "Undefined (tan is undefined when cos(angle) is 0)"

    tan_value = sin_value / cos_value

    return round(tan_value, decimal_places)
def sum(n):
    k = 0
    for i in range(n):
        k += (i + 1)
    return(k)
def calculate_e(terms):
    e_appro = 1
    for i in range(1, terms):
        e_appro += 1 / factorial(i)
    return e_appro
e = calculate_e(10000)

