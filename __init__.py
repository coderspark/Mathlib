from decimal import *
import re
import matplotlib.pyplot as plt
import random

factorials = {0: 1, 1: 1}
def calculate_pi(n):
    getcontext().prec = n + 2  # Set precision (additional 2 digits)
    pi = Decimal(0)
    for k in range(n):
        pi += (Decimal(1) / 16**k) * (
            Decimal(4)/(8*k+1) - Decimal(2)/(8*k+4) -
            Decimal(1)/(8*k+5) - Decimal(1)/(8*k+6)
        )
    return pi

pi =  float(calculate_pi(1000))
def listToString(s):
    str1 = ""
    for ele in s:
        str1 += ele
    # return string
    return str1

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

def sin(n, terms=170, decimal_places=50):
    sin_value = n
    sign = -1
    for i in range(3, terms + 1, 2):
        sin_value += sign * (n ** i) / factorial(i)
        sign *= -1
    rounded_result = round(sin_value, decimal_places)
    return rounded_result

def cos(n, terms=170, decimal_places=12):
    cos_value = 1
    sign = -1
    for i in range(2, terms + 1, 2):
        cos_value += sign * (n ** i) / factorial(i)
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
    if (type(n) == "int"):
        k = 0
        for i in range(n):
            k += (i + 1)
        return(k)
    if (isinstance(n, list)):
        k = 0
        for i in n:
            k += i
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
    result = 0.0
    while n >= base:
        n /= base
        result += 1
    return result

def ln(n):
    return log(n, e)

def equation(functiestr, xn='x', yn='y',yv=3, printb=0):
    functie = re.split(r'([)])',functiestr)
    functiestr = listToString(functie)
    functie = re.split(r'([(])',functiestr)
    functiestr = listToString(functie)
    functie = functiestr.split(' ')

    #Remove spaces and remove yn to turn it into yv
    for i in functie:
        if i == yn:
            functie[functie.index(i)] = yv
        if i == 'e':
            functie[functie.index(i)] = e
        if i == 'pi':
            functie[functie.index(i)] = pi

    for x in range(len(functie)):
        try:
            if functie[x] == '(':
                nf = []
                n = x
                while True:
                    if functie[n] != '(' and functie[n] != ')':
                        nf.append(functie[n])
                    del functie[n]
                    if functie[n] == ')':
                        del functie[n]
                        break
                for i in range(len(nf)):
                    try:
                        if nf[i] == "√" or nf[i] == 'sqrt':
                            nf[i + 1] = sqrt(float(nf[i + 1]))
                            del nf[i]
                    except IndexError:
                        break
                #sin
                for i in range(len(functie)):
                    try:
                        if nf[i] == "sin":
                            nf[i] = sin(float(nf[i + 1]))
                            del nf[i + 1]
                    except IndexError:
                        break
                #cos
                for i in range(len(functie)):
                    try:
                        if nf[i] == "cos":
                            nf[i] = cos(float(nf[i + 1]))
                            del nf[i + 1]
                    except IndexError:
                        break
                #tan
                for i in range(len(functie)):
                    try:
                        if nf[i] == "tan":
                            nf[i] = tan(float(nf[i + 1]))
                            del nf[i + 1]
                    except IndexError:
                        break
                #Exponents
                for i in range(len(nf)):
                    try:
                        if nf[i] == '^':
                                nf[i - 1] = float(nf[i - 1])
                                nf[i + 1] = float(nf[i + 1])
                                for n in range(int(nf[i + 1]) - 1):
                                    nf[i - 1] *= nf[i + 1]
                                del nf[i]
                                del nf[i]
                    except IndexError:
                        break

                #Division
                for i in range(len(nf)):
                    try:
                        if nf[i] == '/':
                            nf[i] = (float(nf[i - 1]) / float(nf[i + 1]))
                            del nf[i - 1]
                            del nf[i]
                    except IndexError:
                        break

                #Multiplication
                for i in range(len(nf)):
                    try:
                        if nf[i] == '*':
                            nf[i] = (float(nf[i - 1]) * float(nf[i + 1]))
                            del nf[i - 1]
                            del nf[i]
                    except IndexError:
                        break
                #Addition and subtraction
                for i in range(len(nf)):
                    try:
                        if nf[i] == '+':
                            nf[i] = (float(nf[i - 1]) + float(nf[i + 1]))
                            del nf[i - 1]
                            del nf[i]
                        if nf[i] == '-':
                            nf[i] = (float(nf[i - 1]) - float(nf[i + 1]))
                            del nf[i - 1]
                            del nf[i]
                        
                    except IndexError:
                        break
                functie.insert(x, nf[0])
        except IndexError:
            break
    #root
    for i in range(len(functie)):
        try:
            if functie[i] == "√" or functie[i] == 'sqrt':
                functie[i] = sqrt(float(functie[i + 1]))
                del functie[i + 1]
        except IndexError:
            break
    #sin
    for i in range(len(functie)):
        try:
            if functie[i] == "sin":
                functie[i] = sin(float(functie[i + 1]))
                del functie[i + 1]
        except IndexError:
            break
    #cos
    for i in range(len(functie)):
        try:
            if functie[i] == "cos":
                functie[i] = cos(float(functie[i + 1]))
                del functie[i + 1]
        except IndexError:
            break
    #tan
    for i in range(len(functie)):
        try:
            if functie[i] == "tan":
                functie[i] = tan(float(functie[i + 1]))
                del functie[i + 1]
        except IndexError:
            break
    #Exponents
    for i in range(len(functie)):
        try:
            if functie[i] == '^':
                functie[i] = (float(functie[i - 1]) ** float(functie[i + 1]))
                del functie[i - 1]
                del functie[i]
        except IndexError:
            break

    #Division
    for i in range(len(functie)):
        try:
            if functie[i] == '/':
                functie[i] = (float(functie[i - 1]) / float(functie[i + 1]))
                del functie[i - 1]
                del functie[i]
        except IndexError:
            break

    #Multiplication
    for i in range(len(functie)):
        try:
            if functie[i] == '*':
                functie[i] = (float(functie[i - 1]) * float(functie[i + 1]))
                del functie[i - 1]
                del functie[i]
        except IndexError:
            break
    #Addition and subtraction
    for i in range(len(functie)):
        try:
            if functie[i] == '+':
                functie[i] = (float(functie[i - 1]) + float(functie[i + 1]))
                del functie[i - 1]
                del functie[i]
            if functie[i] == '-':
                functie[i] = (float(functie[i - 1]) - float(functie[i + 1]))
                del functie[i - 1]
                del functie[i]
        except IndexError:
            break
    xv = functie[len(functie) - 1]
    if printb == 1:
        print(xn, "=", float(xv))
    return float(xv)

# def graph(equation, x_range=(-10, 10), num_points=1000):
#     x_values = np.linspace(x_range[0], x_range[1], num_points)
#     y_values = [function(equation, xn='y', yn='x', yv=x) for x in x_values]
#     plt.figure(figsize=(8, 6))
#     plt.plot(x_values, y_values, label=equation)
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.xlim(min(x_values), max(x_values))
#     plt.ylim(min(y_values), max(y_values))
#     plt.title('Graph of ' + equation)
#     plt.grid(True)
#     plt.legend()
#     plt.show()

class GraphingCalculator:
    def __init__(self, screens):
        self.figures = [None] * screens
        self.ax = [None] * screens
        plt.ion()  # Turn on interactive mode
        
    def draw_graph(self, function, figure_to_draw_in, xRange=(-10, 10), num_points=1000):
        figure_to_draw_in -= 1
        if self.figures[figure_to_draw_in] is None:
            fig = plt.figure()
            ax = fig.gca()
            ax.grid()
            self.figures[figure_to_draw_in] = fig
            self.ax[figure_to_draw_in] = ax
        else:
            fig = self.figures[figure_to_draw_in]
            ax = self.ax[figure_to_draw_in]   
        x = []
        for i in range(num_points):
            x.append(xRange[0] + i * (xRange[1] - xRange[0]) / num_points)
        y = [equation(function, yn='x', xn='y', yv=xi) for xi in x]

        ax.plot(x, y, label=function)
        ax.set_title("Graph of the Function")
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")

        fig.canvas.draw()

class Matrix():
    def __init__(self, m):
        self.m = m
    @classmethod
    def Num(cls, n, length, height):
        x = []
        y = []
        for _ in range(height):
            for _ in range(length):
                x.append(n)
            y.append(x)
            x = []
        return Matrix(y)
    @classmethod
    def Rand(cls, length, height, min=1, max=10):
        x = []
        y = []
        for _ in range(height):
            for _ in range(length):
                x.append(random.randint(min, max))
            y.append(x)
            x = []
        return Matrix(y)
    def rotate(self):
        x = []
        y = []
        for i in range(len(self.m[0])):
            for n in range(len(self.m)):
                x.append(self.m[n][i])
            y.append(x)
            x = []
        self.m = y
        return y
    def inverse(self):
        m = self.m
        if (len(self.m) != len(self.m[0]) or len(self.m) != 2):
            print("Unable to inverse")
            return
        m[0][0], m[1][1] = m[1][1], m[0][0]
        m[0][1] = -m[0][1]
        m[1][0] = -m[1][0]

        return Matrix(m)
    def __str__(self):
        x = ""
        y = ""
        for i in self.m:
            s = 0
            for n in self.m[self.m.index(i)]:
                s += 1
                x += str(n)
                if (s != len(self.m[self.m.index(i)])):
                    x += " "*(5 - len(str(n)))
                    x += " "
            y += str(x)
            x = ""
            if (self.m.index(i) != len(self.m) - 1):
                y += "\n"
        return y
    # choose how it should be converted to a list
    def __iter__(self):
        return iter(self.m)
    def __add__(self, a):
        l = a.m
        x = []
        y = []
        if(len(self.m) == len(l)):
            for i in range(len(self.m)):
                for n in self.m[i]:
                    x.append(n + l[i][self.m[i].index(n)])
                y.append(x)
                x = []
            return Matrix(y)
        else:
            print("Unable to add")
            return
    def __neg__(self):
        x = []
        y = []
        for i in range(len(self.m)):
            for n in range(len(self.m[i])):
                x.append(-self.m[i][n])
            y.append(x)
            x = []
        return Matrix(y)
    def __sub__(self, a):
        l = a.m
        x = []
        y = []
        if(len(self.m) == len(l)):
            for i in range(len(self.m)):
                for n in self.m[i]:
                    x.append(n - l[i][self.m[i].index(n)])
                y.append(x)
                x = []
            return Matrix(y)
        else:
            print("Unable to subtract")
            return
        
    def __mul__(self, a):
        l = a.m   
        if(len(self.m) == len(l)):
            self.rotate()
        # Initialize an empty Matrix to store the result
        result = Matrix.Num(0, len(l[0]), len(self.m)).m

        # Perform Matrix multiplication to calculate the dot product
        for i in range(len(self.m)):
            for j in range(len(l[0])):
                for k in range(len(l)):
                    result[i][j] += self.m[i][k] * l[k][j]
        return Matrix(result)
    def __truediv__(self, a):
        if (len(self.m) == len(self.m[0]) and len(self.m) == 2):
            return self * a.inverse()
        else:
            raise ValueError("Matrix is not a 2x2 matrix")
e = calculate_e(1000)