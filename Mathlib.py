from decimal import *
pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
factorials = {0: 1, 1: 1}
def fibonacci(n, printb=0):
    if n == 1 or n == 2:
        return 1
    if printb == 1:
        x = 0
        y = 1
        print("1: " + str(y))
        for i in range(n - 1):
            z = x + y
            print(str(i + 2) + ": " + str(z))
            x = y
            y = z
        print("\n \n")
        return z
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
def sqrt(n):
    x = n
    count = 0
    while True:
        count += 1
        root = 0.5 * (x + (n / x))
        if (abs(root - x) < 0.000000000000000000000001):
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
    rounded_result = round(sin_value, decimal_places)
    return rounded_result
def cos(n, terms=170, decimal_places=12):
    angle_in_radians = n * (pi / 180.0)
    cos_value = 1
    sign = -1
    for i in range(2, terms + 1, 2):
        cos_value += sign * (angle_in_radians ** i) / factorial(i)
        sign *= -1
    rounded_result = round(cos_value, decimal_places)
    return rounded_result
def tan(n, terms=170, decimal_places=12):
    sin_value = sin(n, terms, decimal_places)
    cos_value = cos(n, terms, decimal_places)
    if cos_value == 0:
        return "Undefined (tan is undefined when cos(angle) is 0)"
    tan_value = sin_value / cos_value
    return round(tan_value, decimal_places)
def sum(n):
    k = 0
    for i in range(n):
        k += (i + 1)
    return(k)
def calculate_e(n_terms):
    getcontext().prec = n_terms + 100  # Set precision to include extra digits
    e_approximation = Decimal(1)
    nfactorial = Decimal(1)
    for n in range(1, n_terms):
        nfactorial *= n
        e_approximation += Decimal(1) / nfactorial
    return e_approximation
def log(n, base=10):
    if n <= 0 or base <= 0 or base == 1:
        raise ValueError("n and base need to be higher than 0 and it cannot be base 1")
    result = 0
    while n >= base:
        n /= base
        result += 1
    return result
def ln(n):
    return log(n, e)

class matrix():
    def __init__(self, m):
        self.m = m
    def add(self, a):
        l = a.m
        x = []
        y = []
        z = []
        if any(isinstance(element, list) for element in self.m):
            for n in range(len(self.m)):
                for i in range(len(self.m[1])):
                    p = self.m[n]
                    pl = l[n]
                    if n == 0:
                        x.append(p[i] + pl[i])
                    elif n == 1:
                        y.append(p[i] + pl[i])
                    else:
                        z.append(p[i] + pl[i])
            f = []
            f.append(x)
            f.append(y)
            if len(self.m) == 3:
                f.append(z)
            return f

        else:
            for i in range(len(self.m)):
                x.append(self.m[i] + l[i])
            return x
    def subtract(self, a):
        l = a.m
        x = []
        y = []
        z = []
        if any(isinstance(element, list) for element in self.m):
            for n in range(len(self.m)):
                for i in range(len(self.m[n])):
                    p = self.m[n]
                    pl = l[n]
                    if n == 0:
                        x.append(p[i] - pl[i])
                    elif n == 1:
                        y.append(p[i] - pl[i])
                    else:
                        z.append(p[i] - pl[i])
            f = []
            f.append(x)
            f.append(y)
            if len(self.m) == 3:
                f.append(z)
            return f

        else:
            for i in range(len(self.m)):
                x.append(self.m[i] - l[i])
            return x
    def multiply(self, a):
        l = a.m
        x = []
        y = []
        z = []
        if any(isinstance(element, list) for element in self.m):
            if len(l) != len(self.m):
                if len(self.m[0]) != len(l):
                    print("Unable to multiply")
                    return
                c = []
                nl = []
                for n in range(len(l[0])):
                    for i in range(len(l)):
                        c.append(l[i][n])
                    nl.append(c)
                    c = []
                l = nl
            for n in range(len(self.m)):
                for i in range(len(l[n])):
                    p = self.m[n]
                    if len(self.m) == 2:
                        if n == 0:
                            x.append((self.m[0][i] * l[0][i]))
                            if len(x) == len(self.m[0]):
                                for n in range(len(x) - 1):
                                    x[0] += x[n + 1]
                                for w in range(1, len(x) - 1):
                                    x.remove(x[w])
                                x.remove(x[1])
                                for _ in range(len(p)):
                                    x.append((self.m[0][_] * l[1][_]))
                                    if len(x) == len(self.m[0]) + 1:
                                        for v in range(2, len(x)):
                                            x[1] += x[v]
                                        for w in range(2, len(x) - 1):
                                            x.remove(x[w])
                                x.remove(x[2])
                        if n == 1:
                            y.append((self.m[1][i] * l[0][i]))
                            if len(y) == len(self.m[0]):
                                for n in range(len(y) - 1):
                                    y[0] += y[n + 1]
                                for w in range(1, len(y) - 1):
                                    y.remove(y[w])
                                y.remove(y[1])
                                for _ in range(len(p)):
                                    y.append((self.m[1][_] * l[1][_]))
                                    if len(y) == len(self.m[0]) + 1:
                                        for v in range(2, len(y)):
                                            y[1] += y[v]
                                        for w in range(2, len(y) - 1):
                                            y.remove(y[w])
                                y.remove(y[2])
                            
                                
            return [x, y]


                     
e = calculate_e(1000)
