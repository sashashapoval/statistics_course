
import numpy as np
def plt_fit(xx, yy, ax, lnwdth=0.8, lnstyle='-', color=''):
    import numpy as np
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    import scipy.stats
    model = LinearRegression(fit_intercept=True)
    #Avoiding the repetition, write a function to draw scatter plot
    #with the regression line
    def regress(x, y):
        X = x[:, np.newaxis]
        model.fit(X, y)
        xfit = np.linspace(np.min(x), np.max(x))
        Xfit = xfit[:, np.newaxis]
        yfit = model.predict(Xfit)
        return model.coef_[0], model.intercept_, xfit, yfit
    
    slope, intercept, xfit1, yfit1 = regress(np.log(xx), np.log(yy))
    yy_pred = np.exp(intercept + slope * np.log(xx))
    anntxt = r'$\sim T^{{{:.2f}}}$'.format(slope)
    if color == '':
        ax.plot(xx, yy_pred, label = anntxt, linewidth = lnwdth, linestyle=lnstyle)
    else:
        ax.plot(xx, yy_pred, label = anntxt, linewidth = lnwdth, linestyle=lnstyle, color=color)

#x[] and y[] must be defined as np.array()
def regress(x, y):
    import numpy as np
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True)
    X = x[:, np.newaxis]
    model.fit(X, y)
    xfit = np.linspace(np.min(x), np.max(x))
    Xfit = xfit[:, np.newaxis]
    yfit = model.predict(Xfit)
    return model.coef_[0], model.intercept_, xfit, yfit

#Modification of regress();
#the lengths along x-values affects the weights at the optimization
#<num> is the number of values in the fit-array
def regress_weigh(x, y, num=100):
    func_name = 'regress_weigh'
    is_err = False
    if len(x) != len(y) or not (len(x) >= 2):
        print(f'{func_name}: Input error\nInput arrays: {x}, {y}')
        return 0, 0, 0, 0, True
    #sort x[] and change y[] accordingly
    sort_index = np.argsort(x)
    x = [x[index] for index in sort_index]
    y = [y[index] for index in sort_index]
    #weights
    delta = []
    delta.append((x[1] - x[0]) / 2)
    for i in range(1, len(x)-1):
        delta.append((x[i+1] - x[i-1]) / 2)
    delta.append((x[-1] - x[-2]) / 2)
    delta_sum = np.sum(delta)
    w = [d / delta_sum for d in delta]
    #computation of sums
    wx = 0
    wy = 0
    wxx = 0
    wxy = 0
    for i in range(len(x)):
        wx += w[i] * x[i]
        wy += w[i] * y[i]
        wxx += w[i] * x[i] * x[i]
        wxy += w[i] * x[i] * y[i]
    #linear equation
    A = np.array([[wx, 1], [wxx, wx]])
    rhs = np.array([wy, wxy])
    sol = np.linalg.solve(A, rhs)
    k = sol[0]
    b = sol[1]
    #fit
    xfit = np.linspace(x[0], x[-1], num=100)
    yfit = [k * xf + b for xf in xfit]
    return k, b, xfit, yfit, is_err

#draw power law function
#with the exponent <exponent>, crossing point <point> at range <range>
def draw_powerlaw(ax, point, exponent, x_range, color='red', lnstyle = '-', linewidth = 1,
                  label = -1, alpha = None):
    c = point[0]**(-exponent) * point[1]
    x = np.linspace(x_range[0], x_range[1])
    y = c/x**(-exponent)
    if label == -1:
        ax.plot(x, y, color=color, linestyle=lnstyle, linewidth=linewidth, alpha=alpha)
    else:
        ax.plot(x, y, color=color, linestyle=lnstyle, linewidth=linewidth, label=label, alpha=alpha)


