import numpy as np
from enum import Enum, StrEnum
import matplotlib.pyplot as plt



class MSColors(StrEnum):
    DARK_BLUE = "#1E3461"
    RED = "#D1453D"
    GREEN = "#5B9276"
    GREY = "#7F7F7F"
    ORANGE = "#E79325"
    LIGHT_BLUE = "#477D9D"
        
if __name__ == '__main__':
    # print(MSColors.GREY)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.plot(x, y, c=MSColors.GREY)  # Utilisation de __str__
    plt.title('Graph avec couleur')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()