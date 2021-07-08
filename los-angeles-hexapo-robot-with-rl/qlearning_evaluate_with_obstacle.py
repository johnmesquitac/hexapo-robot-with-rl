import numpy as np
import pickle
from qlearning_training import initialize_state_matrix, identifiesgoal_state, identifies_state
from matplotlib import pyplot as plt # pylint: disable=import-error
import plotly.express as px # pylint: disable=import-error
import pandas as pd 
from PIL import Image as im
import cv2
action_space = np.array([0, 1, 2, 3])


def identifies_index(state):
    for i in range(enviromentsize):
        for j in range(enviromentsize):
            if state_matrix[i][j] == state:
                return i, j


def identifies_state_matrix(i, j):
    return state_matrix[i][j]
# verificar se tomando uma nova decisão não tomei a mesma anteriormente pra evitar caminhos longos


def next_step(action, state, goal_state):

    if state == goal_state:
        return 10, state
    else:
        i, j = identifies_index(state)
        # up
        if action == 0 and i > 0:
            i -= 1
        # Move left
        elif action == 1 and j > 0:
            j -= 1
        # Move down
        elif action == 2 and i < enviromentsize - 1:
            i += 1
        # Move right
        elif action == 3 and j < enviromentsize - 1:
            j += 1
        steps.append(action)
        next_state = identifies_state_matrix(i, j)
        reward = -1
        return reward, int(next_state)


def select_optimal_action(state):
    optimal = np.argmax(Q[state], axis=0)
    return optimal


def define_steps():
    for step in steps:
        if step == 0:
            steps_desc.append('U')
        elif step == 1:
            steps_desc.append('L')
        elif step == 2:
            steps_desc.append('D')
        elif step == 3:
            steps_desc.append('R')


def select_optimal_path(q_table, enviroment, state_evaluate):
    global steps, steps_desc
    i, j = identifies_state(enviroment, enviromentsize)
    k, l = identifiesgoal_state(enviroment, enviromentsize)
    state = int(state_matrix[i][j])
    goal_state = int(state_matrix[k][l])
    states = []
    steps = []
    steps_desc = []
    states.append(state)
    reward, next_state = 0, 0
    done = False
    while(not done):
        print(state)
        action = select_optimal_action(state)
        reward, next_state = next_step(action, state, goal_state)
        state = next_state
        states.append(state)
        if reward == 10:
            done = True
    states = states[:-1]
    define_steps()
    print(q_table, '\n', '\n', states, '\n', steps, '\n', steps_desc, '\n', 'path_length:', len(states))
    plot_q_with_steps(enviroment, states,enviromentsize, state_evaluate)
    steps_matrix.append(steps_desc)

def plot_q_with_steps(enviroment, steps, enviromentsize, state):
    for step in steps:
        if step!=state:
            i,j = identifies_index(step)
            enviroment[i][j] = 5
    plot_matrix(enviroment,enviromentsize,enviromentsize,state)


def reset_enviroment(enviroment, env_size, goal_position, obstacles_position):
    enviroment = np.zeros((env_size, env_size))
    i, j = identifies_state_train(goal_position, env_size)
    enviroment[i][j] = 20    
    for obstacle in obstacles_position:
        i,j = identifies_state_train(obstacle, env_size)
        enviroment[i][j] = -1
    enviroment[0][0] = 1
    return enviroment


def identifies_state_train(goal_position, size):
    for i in range(size):
        for j in range(size):
            if state_matrix[i][j] == goal_position:
                return i, j

def plot_matrix(matrix, x_size, y_size, state):
    cmap = plt.cm.copper
    plt.imshow(matrix, cmap=cmap)
    plt.title('Best Path to reach state '+str(state)+' in Enviroment')
    plt.tight_layout()
    plt.savefig("imgs/with_obstacles/Q/Q"+str(state)+"evaluate.png")
    '''    array = np.reshape(matrix, (x_size, y_size))
        array = array.astype(np.uint8)
        data= im.fromarray(array)
        new_data = data.convert('L')
        new_data.save('imgs/with_obstacles/'+str(state)+".png")
        image = cv2.imread('imgs/with_obstacles/'+str(state)+".png")
        image = ~image
        cv2.imwrite('imgs/with_obstacles/'+str(state)+".png", image)'''

def main():
    global state_matrix, enviromentsize, Q, steps_matrix
    steps_matrix = []
    enviromentsize = 49
    print('begin')
    state_matrix = initialize_state_matrix(
        np.zeros((enviromentsize, enviromentsize)), enviromentsize)
    obstacles =  [250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 264, 265, 266, 267, 268, 269, 270, 271, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 362, 363, 364, 365, 366, 367, 368, 369, 370, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 411, 412, 413, 414, 415, 416, 417, 418, 419, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 447, 465, 466, 467, 468, 470, 471, 472, 473, 477, 478, 479, 480, 481, 482, 483, 514, 515, 516, 527, 528, 529, 530, 531, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 560, 561, 562, 563, 564, 565, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 609, 610, 611, 612, 613, 614, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 658, 659, 660, 661, 662, 663, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 692, 693, 694, 698, 699, 700, 701, 702, 707, 708, 709, 710, 711, 712, 716, 717, 718, 719, 725, 726, 727, 728, 756, 757, 758, 759, 760, 761, 766, 805, 806, 807, 808, 809, 810, 814, 815, 816, 838, 839, 840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 857, 858, 859, 863, 864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 875, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 906, 907, 908, 912, 913, 914, 915, 916, 917, 918, 919, 920, 921, 922, 923, 924, 936, 937, 938, 939, 940, 941, 942, 943, 944, 945, 946, 947, 948, 950, 951, 952, 953, 954, 955, 956, 957, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 999, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1053, 1054, 1055, 1056, 1057, 1058, 1065, 1066, 1067, 1068, 1069, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1099, 1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107, 1113, 1114, 1115, 1116, 1117, 1118, 1132, 1133, 1134, 1135, 1136, 1137, 1138, 1139, 1140, 1141, 1142, 1143, 1148, 1149, 1150, 1160, 1161, 1162, 1163, 1164, 1165, 1166, 1167, 1168, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1197, 1198, 1199, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1246, 1247, 1248, 1295, 1296, 1297, 1305, 1306, 1307, 1308, 1309, 1315, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1344, 1345, 1346, 1347, 1348, 1349, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1393, 1394, 1395, 1396, 1397, 1398, 1403, 1404, 1405, 1406, 1407, 1408, 1409, 1410, 1411, 1412, 1413, 1414, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1433, 1434, 1435, 1442, 1443, 1444, 1445, 1446, 1447, 1452, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1491, 1492, 1493, 1494, 1495, 1496, 1540, 1541, 1542, 1543, 1544, 1545, 1560, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1589, 1590, 1591, 1592, 1593, 1594, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1638, 1639, 1640, 1641, 1642, 1643, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1681, 1682, 1687, 1688, 1689, 1690, 1691, 1692, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1721, 1722, 1723, 1724, 1725, 1726, 1727, 1728, 1729, 1730, 1736, 1737, 1738, 1739, 1740, 1741, 1818, 1819, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1834, 1835, 1836, 1837, 1838, 1839, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1883, 1884, 1885, 1886, 1887, 1888, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1932, 1933, 1934, 1935, 1936, 1937, 1942, 1943, 1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1981, 1982, 1983, 1984, 1985, 1986, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2015, 2029, 2030, 2033, 2034, 2035, 2040, 2041, 2042, 2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2063, 2064, 2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2082, 2083, 2084, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097, 2098, 2099, 2100, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2131, 2132, 2133, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2180, 2181, 2182, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2230, 2231, 2236, 2237, 2238, 2239, 2240, 2241, 2242, 2243, 2244, 2245, 2246, 2247, 2286, 2287, 2288, 2289, 2290, 2291, 2292, 2293, 2294, 2295, 2296, 2308, 2309, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2334, 2335, 2336, 2337, 2338, 2339, 2340, 2341, 2342, 2343, 2344, 2345, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2364, 2365, 2366, 2367, 2368, 2373, 2374, 2375, 2383, 2384, 2385, 2386, 2387, 2388, 2389, 2390, 2391, 2392, 2393, 2394]
    i = 2400
    state_matrix = initialize_state_matrix(
                np.zeros((enviromentsize, enviromentsize)), enviromentsize)
    env = np.zeros((enviromentsize, enviromentsize))
    env = reset_enviroment(env, enviromentsize, i, obstacles)
    with open(r'C:\Users\mesqu\Downloads\TG\hexapo-robot-optmal\pickle\2401.pickle', "rb") as read:
        Q = pickle.load(read)
    print('evaluating optimal path to position:',i)
    select_optimal_path(Q, env, i)

    print('steps', steps_matrix)

    with open('pickle/with_obstacles/steps_positions.pickle', "wb") as write:
        pickle.dump(steps_matrix, write)


if __name__ == '__main__':
    main()
