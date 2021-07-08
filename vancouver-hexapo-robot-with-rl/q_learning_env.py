import math
import numpy as np
import random

class enviroment:
    def __init__(self):
        self.x = 101
        self.y = 101
        self.cont = 0
        self.x_position = 0
        self.y_position = 0
        self.x_old_state = 0
        self.y_old_state = 0
        self.goalx = self.x-1
        self.goaly = self.y-1
        self.action_space = [0,1,2,3]
        self.obstacles_map = [11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34, 35, 49, 56, 58, 59, 75, 76, 77, 78, 79, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 113, 114, 115, 116, 117, 123, 124, 125, 126, 127, 128, 129, 132, 133, 134, 135, 136, 137, 159, 177, 178, 179, 180, 181, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 215, 216, 217, 224, 225, 226, 227, 228, 229, 232, 233, 234, 235, 236, 237, 238, 244, 279, 280, 281, 282, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 302, 317, 322, 323, 326, 327, 328, 329, 332, 333, 334, 335, 336, 337, 338, 344, 345, 346, 381, 382, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 422, 423, 424, 425, 428, 429, 432, 433, 434, 435, 436, 437, 438, 439, 445, 446, 447, 448, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 522, 523, 524, 525, 526, 527, 532, 533, 534, 535, 536, 537, 538, 539, 546, 547, 548, 549, 550, 588, 589, 596, 597, 598, 599, 600, 601, 602, 603, 604, 622, 623, 624, 625, 626, 627, 628, 629, 632, 633, 634, 635, 636, 637, 638, 639, 648, 649, 650, 651, 688, 689, 690, 691, 699, 700, 701, 702, 703, 704, 724, 725, 726, 727, 728, 729, 734, 735, 736, 737, 738, 750, 751, 788, 789, 790, 791, 792, 793, 801, 802, 803, 804, 826, 827, 828, 829, 830, 831, 836, 837, 838, 866, 867, 868, 888, 889, 890, 891, 892, 893, 894, 895, 928, 929, 930, 931, 932, 938, 945, 946, 967, 968, 969, 970, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997, 1023, 1024, 1030, 1031, 1032, 1037, 1045, 1046, 1047, 1048, 1067, 1068, 1069, 1070, 1071, 1072, 1088, 1089, 1090, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1099, 1123, 1124, 1125, 1126, 1132, 1133, 1134, 1137, 1138, 1139, 1145, 1146, 1147, 1148, 1149, 1150, 1168, 1169, 1170, 1171, 1172, 1173, 1176, 1177, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1223, 1224, 1225, 1226, 1227, 1228, 1234, 1235, 1236, 1239, 1245, 1246, 1247, 1248, 1249, 1250, 1251, 1270, 1271, 1272, 1273, 1276, 1277, 1278, 1279, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1336, 1337, 1338, 1339, 1346, 1347, 1348, 1349, 1350, 1351, 1366, 1372, 1373, 1376, 1377, 1378, 1379, 1380, 1381, 1388, 1389, 1390, 1391, 1392, 1393, 1394, 1395, 1396, 1397, 1398, 1399, 1400, 1401, 1402, 1403, 1404, 1405, 1423, 1425, 1426, 1427, 1428, 1429, 1430, 1431, 1432, 1438, 1439, 1444, 1445, 1448, 1449, 1450, 1451, 1466, 1467, 1468, 1476, 1477, 1478, 1479, 1480, 1481, 1482, 1488, 1489, 1490, 1491, 1492, 1493, 1494, 1495, 1496, 1497, 1498, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1523, 1524, 1525, 1527, 1528, 1529, 1530, 1531, 1532, 1533, 1534, 1544, 1545, 1546, 1547, 1550, 1551, 1566, 1567, 1568, 1569, 1570, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1623, 1624, 1625, 1626, 1627, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1644, 1645, 1646, 1654, 1655, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1678, 1679, 1680, 1681, 1682, 1689, 1690, 1691, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1711, 1723, 1724, 1725, 1726, 1727, 1728, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1744, 1745, 1746, 1754, 1755, 1756, 1757, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1773, 1780, 1781, 1782, 1790, 1791, 1792, 1793, 1794, 1797, 1798, 1799, 1800, 1801, 1802, 1803, 1804, 1805, 1811, 1812, 1813, 1824, 1825, 1826, 1827, 1828, 1831, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1846, 1847, 1848, 1849, 1855, 1856, 1857, 1858, 1859, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1875, 1876, 1882, 1888, 1892, 1893, 1894, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1911, 1912, 1913, 1914, 1915, 1919, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1935, 1936, 1937, 1938, 1939, 1948, 1949, 1950, 1951, 1955, 1957, 1958, 1959, 1960, 1961, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1975, 1976, 1977, 1978, 1988, 1989, 1990, 1994, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2020, 2021, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2037, 2038, 2039, 2050, 2051, 2052, 2055, 2059, 2060, 2061, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2076, 2077, 2078, 2079, 2080, 2088, 2089, 2090, 2091, 2092, 2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2122, 2123, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2145, 2146, 2152, 2153, 2154, 2155, 2156, 2161, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2188, 2189, 2190, 2191, 2192, 2193, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218, 2222, 2223, 2224, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2245, 2246, 2247, 2248, 2254, 2255, 2256, 2257, 2266, 2267, 2268, 2269, 2270, 2271, 2272, 2273, 2276, 2277, 2278, 2279, 2280, 2281, 2282, 2283, 2290, 2291, 2292, 2293, 2298, 2299, 2300, 2301, 2302, 2303, 2304, 2305, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2323, 2324, 2334, 2335, 2336, 2337, 2338, 2339, 2345, 2346, 2347, 2348, 2349, 2350, 2356, 2357, 2366, 2367, 2368, 2369, 2370, 2371, 2372, 2373, 2376, 2377, 2378, 2379, 2380, 2381, 2382, 2383, 2392, 2393, 2394, 2398, 2399, 2400, 2401, 2402, 2403, 2404, 2405, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2428, 2429, 2436, 2437, 2438, 2439, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2483, 2488, 2494, 2495, 2498, 2499, 2500, 2501, 2502, 2503, 2504, 2505, 2511, 2512, 2513, 2514, 2515, 2516, 2517, 2518, 2519, 2520, 2521, 2522, 2523, 2524, 2528, 2529, 2530, 2531, 2538, 2539, 2545, 2547, 2548, 2549, 2550, 2551, 2553, 2554, 2566, 2567, 2569, 2570, 2571, 2572, 2573, 2576, 2577, 2578, 2579, 2580, 2581, 2582, 2583, 2588, 2589, 2590, 2598, 2599, 2600, 2601, 2602, 2603, 2604, 2605, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2628, 2629, 2630, 2631, 2632, 2633, 2645, 2646, 2647, 2649, 2650, 2651, 2654, 2655, 2656, 2666, 2667, 2668, 2669, 2671, 2672, 2673, 2676, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2688, 2689, 2690, 2691, 2692, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2745, 2746, 2747, 2748, 2749, 2752, 2754, 2755, 2756, 2757, 2758, 2766, 2767, 2768, 2769, 2770, 2771, 2776, 2777, 2778, 2779, 2780, 2781, 2782, 2783, 2788, 2789, 2790, 2791, 2792, 2793, 2794, 2800, 2801, 2802, 2803, 2804, 2805, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824, 2825, 2826, 2827, 2828, 2831, 2832, 2833, 2834, 2835, 2836, 2837, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2854, 2855, 2856, 2857, 2858, 2859, 2860, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2876, 2877, 2878, 2879, 2880, 2881, 2882, 2883, 2888, 2889, 2890, 2891, 2892, 2893, 2894, 2895, 2902, 2903, 2904, 2905, 2911, 2912, 2913, 2914, 2915, 2916, 2917, 2918, 2919, 2920, 2921, 2922, 2923, 2924, 2925, 2926, 2929, 2930, 2932, 2933, 2934, 2935, 2936, 2937, 2938, 2939, 2946, 2947, 2948, 2949, 2950, 2951, 2952, 2953, 2956, 2957, 2958, 2959, 2960, 2961, 2968, 2969, 2970, 2971, 2972, 2973, 2974, 2975, 2977, 2978, 2979, 2980, 2981, 2982, 2983, 2988, 2989, 2990, 2991, 2992, 2993, 2994, 2995, 2998, 2999, 3004, 3005, 3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3027, 3030, 3031, 3032, 3035, 3036, 3037, 3038, 3039, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3058, 3059, 3060, 3061, 3070, 3071, 3072, 3073, 3074, 3075, 3076, 3077, 3080, 3081, 3082, 3083, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3098, 3099, 3100, 3101, 3111, 3112, 3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125, 3126, 3127, 3128, 3131, 3132, 3133, 3134, 3137, 3138, 3139, 3150, 3151, 3152, 3153, 3154, 3155, 3156, 3160, 3161, 3172, 3173, 3174, 3175, 3176, 3177, 3178, 3182, 3188, 3189, 3190, 3191, 3192, 3193, 3194, 3195, 3198, 3199, 3200, 3201, 3202, 3203, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223, 3224, 3225, 3226, 3227, 3228, 3232, 3233, 3234, 3235, 3236, 3238, 3239, 3245, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3267, 3268, 3274, 3275, 3276, 3277, 3278, 3288, 3289, 3290, 3291, 3292, 3293, 3294, 3295, 3298, 3299, 3300, 3301, 3302, 3303, 3304, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3333, 3334, 3335, 3336, 3337, 3338, 3345, 3346, 3347, 3354, 3355, 3356, 3357, 3358, 3359, 3360, 3361, 3367, 3368, 3369, 3370, 3376, 3377, 3378, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3398, 3399, 3400, 3401, 3402, 3403, 3404, 3405, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3445, 3446, 3447, 3448, 3449, 3456, 3457, 3458, 3459, 3460, 3461, 3467, 3468, 3469, 3470, 3471, 3472, 3478, 3488, 3489, 3490, 3491, 3492, 3493, 3494, 3495, 3498, 3499, 3500, 3501, 3502, 3503, 3504, 3505, 3515, 3516, 3517, 3518, 3519, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3533, 3534, 3535, 3536, 3537, 3538, 3539, 3545, 3546, 3547, 3548, 3549, 3550, 3551, 3559, 3560, 3561, 3569, 3570, 3571, 3572, 3573, 3574, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3611, 3617, 3618, 3619, 3620, 3621, 3622, 3623, 3624, 3625, 3626, 3627, 3628, 3634, 3635, 3637, 3638, 3639, 3647, 3648, 3649, 3650, 3651, 3652, 3653, 3660, 3661, 3667, 3671, 3673, 3674, 3675, 3676, 3688, 3689, 3691, 3692, 3693, 3694, 3695, 3698, 3699, 3700, 3701, 3702, 3703, 3704, 3705, 3711, 3712, 3713, 3719, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3727, 3728, 3733, 3735, 3736, 3745, 3746, 3749, 3750, 3751, 3752, 3753, 3754, 3755, 3767, 3768, 3769, 3775, 3776, 3777, 3778, 3788, 3789, 3790, 3791, 3794, 3795, 3798, 3799, 3800, 3801, 3802, 3803, 3804, 3805, 3811, 3812, 3813, 3814, 3815, 3821, 3822, 3823, 3824, 3825, 3826, 3827, 3828, 3833, 3834, 3835, 3845, 3846, 3847, 3848, 3851, 3852, 3853, 3854, 3855, 3856, 3857, 3867, 3868, 3869, 3870, 3871, 3877, 3878, 3879, 3880, 3888, 3889, 3890, 3891, 3892, 3893, 3899, 3900, 3901, 3902, 3903, 3904, 3905, 3911, 3912, 3913, 3914, 3915, 3916, 3917, 3922, 3923, 3924, 3925, 3926, 3927, 3928, 3933, 3934, 3935, 3936, 3937, 3945, 3946, 3947, 3948, 3949, 3950, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3978, 3979, 3980, 3981, 3982, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4024, 4025, 4026, 4027, 4028, 4034, 4035, 4036, 4037, 4038, 4039, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4068, 4069, 4070, 4071, 4072, 4073, 4074, 4075, 4080, 4081, 4082, 4083, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4100, 4101, 4102, 4103, 4104, 4105, 4111, 4112, 4113, 4114, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4126, 4127, 4128, 4136, 4137, 4138, 4139, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4157, 4158, 4159, 4160, 4170, 4171, 4172, 4173, 4174, 4175, 4176, 4182, 4192, 4193, 4194, 4195, 4196, 4197, 4198, 4199, 4202, 4203, 4204, 4205, 4211, 4212, 4213, 4214, 4215, 4216, 4217, 4218, 4219, 4220, 4221, 4222, 4223, 4234, 4235, 4238, 4239, 4242, 4243, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4259, 4260, 4272, 4273, 4274, 4275, 4276, 4277, 4289, 4294, 4295, 4296, 4297, 4298, 4299, 4300, 4301, 4304, 4305, 4311, 4312, 4313, 4314, 4315, 4316, 4317, 4318, 4319, 4320, 4321, 4322, 4323, 4324, 4325, 4334, 4335, 4336, 4337, 4343, 4344, 4345, 4351, 4352, 4353, 4354, 4355, 4356, 4357, 4358, 4367, 4368, 4374, 4375, 4376, 4377, 4378, 4379, 4389, 4390, 4391, 4396, 4397, 4398, 4399, 4400, 4401, 4402, 4403, 4411, 4412, 4413, 4414, 4415, 4416, 4417, 4418, 4419, 4420, 4421, 4422, 4423, 4424, 4425, 4426, 4434, 4435, 4436, 4437, 4438, 4439, 4444, 4445, 4446, 4447, 4448, 4453, 4454, 4455, 4456, 4457, 4458, 4459, 4460, 4467, 4468, 4469, 4470, 4477, 4478, 4479, 4480, 4481, 4489, 4490, 4491, 4492, 4493, 4498, 4499, 4500, 4501, 4502, 4503, 4504, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4518, 4519, 4520, 4521, 4522, 4523, 4524, 4525, 4526, 4527, 4536, 4537, 4538, 4539, 4540, 4545, 4546, 4547, 4548, 4549, 4550, 4556, 4557, 4558, 4559, 4560, 4567, 4568, 4569, 4570, 4571, 4572, 4579, 4580, 4581, 4582, 4583, 4589, 4590, 4591, 4592, 4593, 4594, 4595, 4601, 4602, 4603, 4604, 4611, 4612, 4613, 4614, 4615, 4616, 4617, 4618, 4619, 4620, 4621, 4622, 4623, 4624, 4625, 4626, 4627, 4638, 4639, 4640, 4649, 4650, 4651, 4652, 4658, 4659, 4660, 4661, 4667, 4668, 4669, 4670, 4671, 4672, 4673, 4674, 4681, 4682, 4683, 4690, 4691, 4692, 4693, 4694, 4695, 4696, 4697, 4703, 4704, 4712, 4713, 4714, 4715, 4716, 4717, 4718, 4720, 4721, 4722, 4723, 4724, 4725, 4726, 4727, 4740, 4744, 4746, 4753, 4754, 4760, 4761, 4762, 4767, 4768, 4769, 4770, 4771, 4772, 4773, 4774, 4775, 4776, 4783, 4789, 4792, 4793, 4794, 4795, 4796, 4797, 4798, 4799, 4814, 4815, 4816, 4817, 4818, 4821, 4822, 4823, 4824, 4825, 4826, 4827, 4833, 4834, 4844, 4845, 4846, 4847, 4863, 4864, 4867, 4868, 4869, 4870, 4871, 4872, 4873, 4874, 4875, 4876, 4877, 4878, 4889, 4890, 4891, 4894, 4895, 4896, 4897, 4898, 4899, 4900, 4901, 4912, 4913, 4916, 4917, 4918, 4921, 4922, 4923, 4924, 4925, 4926, 4927, 4933, 4934, 4935, 4936, 4944, 4945, 4946, 4947, 4948, 4965, 4966, 4967, 4968, 4969, 4970, 4971, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 4980, 4989, 4990, 4991, 4992, 4993, 4996, 4997, 4998, 4999, 5000, 5001, 5002, 5003, 5012, 5013, 5014, 5015, 5018, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5033, 5034, 5035, 5036, 5037, 5038, 5039, 5045, 5046, 5047, 5048, 5049, 5050, 5051, 5052, 5066, 5067, 5068, 5069, 5070, 5071, 5072, 5073, 5074, 5075, 5076, 5077, 5078, 5079, 5080, 5081, 5082, 5089, 5090, 5091, 5092, 5093, 5094, 5095, 5098, 5099, 5100, 5101, 5102, 5103, 5104, 5112, 5113, 5114, 5115, 5116, 5117, 5121, 5122, 5123, 5124, 5125, 5126, 5127, 5133, 5134, 5135, 5136, 5137, 5138, 5139, 5147, 5148, 5149, 5151, 5152, 5153, 5154, 5155, 5156, 5157, 5168, 5169, 5170, 5171, 5172, 5173, 5174, 5175, 5176, 5177, 5178, 5179, 5180, 5181, 5182, 5183, 5191, 5192, 5193, 5194, 5195, 5196, 5197, 5200, 5201, 5202, 5203, 5204, 5205, 5214, 5215, 5216, 5217, 5218, 5219, 5222, 5223, 5224, 5225, 5226, 5227, 5233, 5234, 5235, 5236, 5237, 5238, 5239, 5252, 5253, 5254, 5255, 5256, 5257, 5258, 5259, 5260, 5261, 5262, 5270, 5271, 5272, 5273, 5274, 5275, 5276, 5277, 5278, 5279, 5280, 5281, 5282, 5293, 5294, 5295, 5296, 5297, 5298, 5299, 5303, 5304, 5305, 5316, 5317, 5318, 5319, 5320, 5321, 5324, 5325, 5326, 5327, 5333, 5334, 5335, 5336, 5337, 5338, 5339, 5343, 5344, 5353, 5354, 5355, 5356, 5357, 5358, 5359, 5360, 5361, 5362, 5363, 5364, 5365, 5366, 5372, 5373, 5374, 5375, 5376, 5377, 5378, 5379, 5380, 5381, 5382, 5396, 5397, 5398, 5399, 5400, 5401, 5405, 5418, 5419, 5420, 5421, 5422, 5423, 5426, 5427, 5433, 5434, 5435, 5436, 5437, 5438, 5439, 5443, 5444, 5445, 5446, 5454, 5455, 5456, 5457, 5458, 5459, 5460, 5461, 5462, 5463, 5464, 5465, 5466, 5467, 5474, 5475, 5476, 5477, 5478, 5479, 5480, 5481, 5482, 5497, 5498, 5499, 5500, 5501, 5502, 5503, 5512, 5513, 5519, 5520, 5521, 5522, 5523, 5524, 5525, 5526, 5533, 5534, 5535, 5536, 5537, 5538, 5539, 5543, 5544, 5545, 5546, 5547, 5548, 5557, 5558, 5559, 5560, 5561, 5562, 5563, 5564, 5565, 5566, 5567, 5568, 5576, 5577, 5578, 5579, 5580, 5581, 5582, 5599, 5600, 5601, 5602, 5603, 5604, 5612, 5613, 5614, 5615, 5622, 5623, 5624, 5625, 5626, 5627, 5633, 5634, 5635, 5636, 5637, 5638, 5639, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5662, 5663, 5664, 5665, 5666, 5667, 5668, 5678, 5679, 5680, 5681, 5682, 5690, 5691, 5694, 5701, 5702, 5703, 5704, 5712, 5713, 5715, 5716, 5717, 5724, 5725, 5726, 5727, 5733, 5734, 5735, 5736, 5737, 5738, 5739, 5740, 5743, 5744, 5745, 5746, 5747, 5748, 5749, 5766, 5767, 5768, 5769, 5780, 5781, 5782, 5791, 5793, 5794, 5795, 5796, 5803, 5804, 5813, 5814, 5817, 5818, 5819, 5826, 5827, 5835, 5836, 5837, 5838, 5839, 5840, 5843, 5844, 5845, 5846, 5847, 5848, 5849, 5882, 5891, 5893, 5894, 5895, 5896, 5915, 5916, 5917, 5919, 5920, 5933, 5937, 5938, 5939, 5943, 5944, 5945, 5946, 5947, 5948, 5949, 5955, 5956, 5957, 5989, 5990, 5991, 5994, 5995, 5996, 6012, 6013, 6014, 6017, 6020, 6021, 6033, 6034, 6035, 6039, 6043, 6044, 6045, 6046, 6047, 6048, 6049, 6055, 6056, 6057, 6058, 6059, 6066, 6067, 6088, 6089, 6090, 6091, 6092, 6096, 6101, 6102, 6112, 6113, 6114, 6120, 6121, 6122, 6123, 6124, 6125, 6133, 6134, 6135, 6136, 6137, 6138, 6143, 6144, 6145, 6146, 6147, 6148, 6149, 6155, 6156, 6157, 6158, 6159, 6160, 6179, 6189, 6190, 6191, 6192, 6193, 6194, 6201, 6202, 6203, 6204, 6212, 6213, 6214, 6222, 6223, 6224, 6225, 6226, 6227, 6235, 6236, 6237, 6238, 6239, 6244, 6245, 6246, 6247, 6248, 6249, 6256, 6257, 6258, 6259, 6260, 6261, 6279, 6280, 6281, 6290, 6291, 6292, 6293, 6294, 6300, 6301, 6302, 6303, 6304, 6305, 6313, 6314, 6315, 6317, 6318, 6319, 6324, 6325, 6326, 6327, 6328, 6337, 6338, 6339, 6346, 6347, 6348, 6349, 6354, 6355, 6358, 6359, 6360, 6361, 6362, 6380, 6381, 6382, 6383, 6384, 6392, 6393, 6394, 6397, 6398, 6399, 6402, 6403, 6404, 6405, 6417, 6418, 6419, 6420, 6421, 6425, 6426, 6427, 6428, 6439, 6442, 6443, 6444, 6448, 6449, 6454, 6455, 6456, 6457, 6460, 6461, 6462, 6481, 6482, 6483, 6484, 6485, 6486, 6494, 6495, 6497, 6498, 6499, 6500, 6501, 6504, 6505, 6518, 6519, 6520, 6521, 6522, 6523, 6524, 6527, 6528, 6533, 6542, 6543, 6544, 6545, 6546, 6554, 6555, 6556, 6557, 6558, 6559, 6582, 6583, 6584, 6585, 6586, 6587, 6588, 6597, 6598, 6599, 6600, 6601, 6602, 6603, 6612, 6620, 6621, 6622, 6623, 6624, 6625, 6626, 6633, 6634, 6635, 6642, 6643, 6644, 6645, 6646, 6647, 6648, 6654, 6655, 6656, 6657, 6658, 6659, 6660, 6661, 6665, 6682, 6683, 6684, 6685, 6686, 6687, 6688, 6689, 6690, 6698, 6699, 6700, 6701, 6702, 6703, 6704, 6712, 6713, 6714, 6722, 6723, 6724, 6725, 6726, 6727, 6733, 6734, 6735, 6736, 6737, 6744, 6745, 6746, 6747, 6748, 6749, 6754, 6755, 6756, 6757, 6758, 6759, 6760, 6761, 6762, 6763, 6783, 6784, 6785, 6786, 6787, 6788, 6789, 6790, 6791, 6792, 6800, 6801, 6802, 6803, 6804, 6805, 6812, 6813, 6814, 6815, 6816, 6824, 6825, 6826, 6827, 6833, 6834, 6835, 6836, 6837, 6846, 6847, 6848, 6849, 6855, 6856, 6857, 6858, 6859, 6860, 6861, 6862, 6863, 6864, 6865, 6866, 6885, 6886, 6887, 6888, 6889, 6890, 6891, 6892, 6893, 6894, 6902, 6903, 6904, 6905, 6912, 6913, 6914, 6915, 6916, 6917, 6918, 6925, 6926, 6927, 6935, 6936, 6937, 6948, 6949, 6957, 6958, 6959, 6960, 6961, 6962, 6963, 6964, 6965, 6966, 6967, 6988, 6989, 6990, 6991, 6992, 6993, 6994, 6995, 6996, 7004, 7005, 7011, 7013, 7014, 7015, 7016, 7017, 7018, 7019, 7020, 7027, 7037, 7043, 7056, 7059, 7060, 7061, 7062, 7063, 7064, 7065, 7066, 7067, 7068, 7090, 7091, 7092, 7093, 7094, 7095, 7096, 7097, 7098, 7111, 7112, 7113, 7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122, 7133, 7134, 7135, 7143, 7144, 7145, 7156, 7157, 7158, 7161, 7162, 7163, 7164, 7165, 7166, 7167, 7168, 7169, 7191, 7192, 7193, 7194, 7195, 7196, 7197, 7198, 7199, 7211, 7212, 7213, 7214, 7215, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7233, 7234, 7235, 7236, 7237, 7243, 7244, 7245, 7246, 7247, 7256, 7257, 7258, 7259, 7260, 7263, 7264, 7265, 7266, 7267, 7268, 7269, 7270, 7292, 7293, 7294, 7295, 7296, 7297, 7298, 7299, 7300, 7301, 7302, 7311, 7312, 7313, 7314, 7315, 7316, 7317, 7319, 7320, 7321, 7322, 7323, 7324, 7325, 7326, 7334, 7335, 7336, 7337, 7338, 7339, 7343, 7344, 7345, 7346, 7347, 7348, 7349, 7356, 7357, 7358, 7359, 7360, 7361, 7362, 7365, 7366, 7367, 7368, 7369, 7370, 7371, 7392, 7393, 7394, 7395, 7396, 7397, 7398, 7399, 7400, 7401, 7402, 7403, 7412, 7413, 7414, 7415, 7416, 7417, 7418, 7421, 7422, 7423, 7424, 7425, 7426, 7427, 7428, 7435, 7436, 7437, 7438, 7439, 7440, 7441, 7445, 7446, 7447, 7448, 7449, 7458, 7459, 7460, 7461, 7462, 7463, 7464, 7467, 7468, 7469, 7470, 7492, 7493, 7495, 7496, 7498, 7499, 7500, 7501, 7502, 7503, 7504, 7514, 7515, 7516, 7517, 7518, 7520, 7521, 7523, 7524, 7525, 7526, 7527, 7528, 7537, 7538, 7539, 7540, 7541, 7542, 7543, 7547, 7548, 7549, 7560, 7561, 7562, 7563, 7564, 7565, 7566, 7569, 7570, 7571, 7592, 7593, 7596, 7597, 7600, 7603, 7604, 7605, 7606, 7616, 7617, 7618, 7620, 7621, 7622, 7623, 7625, 7626, 7627, 7628, 7639, 7640, 7641, 7642, 7643, 7644, 7645, 7649, 7655, 7656, 7662, 7663, 7665, 7666, 7671, 7692, 7698, 7699, 7705, 7706, 7707, 7708, 7710, 7711, 7718, 7720, 7721, 7722, 7723, 7724, 7725, 7727, 7728, 7733, 7734, 7741, 7742, 7743, 7744, 7745, 7746, 7747, 7748, 7755, 7756, 7757, 7758, 7764, 7791, 7792, 7805, 7806, 7807, 7808, 7809, 7810, 7811, 7812, 7813, 7821, 7822, 7823, 7824, 7825, 7826, 7827, 7833, 7834, 7835, 7836, 7843, 7844, 7845, 7846, 7847, 7848, 7849, 7855, 7856, 7857, 7858, 7859, 7860, 7891, 7905, 7906, 7907, 7908, 7909, 7910, 7911, 7912, 7913, 7914, 7915, 7923, 7924, 7925, 7926, 7927, 7928, 7933, 7934, 7935, 7936, 7937, 7938, 7945, 7946, 7947, 7948, 7949, 7950, 7955, 7956, 7957, 7958, 7959, 7960, 7961, 7991, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010, 8011, 8012, 8013, 8014, 8015, 8016, 8017, 8025, 8026, 8027, 8028, 8034, 8035, 8036, 8037, 8038, 8039, 8040, 8047, 8048, 8049, 8050, 8057, 8058, 8059, 8060, 8061, 8062, 8063, 8090, 8091, 8103, 8104, 8105, 8106, 8107, 8108, 8109, 8110, 8111, 8112, 8113, 8114, 8115, 8116, 8117, 8118, 8127, 8128, 8133, 8136, 8137, 8138, 8139, 8140, 8141, 8142, 8149, 8150, 8155, 8156, 8159, 8160, 8161, 8162, 8163, 8164, 8165, 8180, 8190, 8204, 8205, 8206, 8207, 8208, 8209, 8210, 8211, 8212, 8213, 8214, 8215, 8216, 8217, 8218, 8219, 8220, 8221, 8233, 8234, 8235, 8238, 8239, 8240, 8241, 8242, 8243, 8244, 8255, 8256, 8257, 8258, 8261, 8262, 8263, 8264, 8265, 8266, 8280, 8281, 8289, 8290, 8305, 8306, 8307, 8308, 8309, 8310, 8311, 8312, 8313, 8314, 8315, 8316, 8317, 8318, 8319, 8320, 8321, 8322, 8323, 8324, 8333, 8334, 8335, 8336, 8340, 8341, 8342, 8343, 8344, 8345, 8346, 8355, 8356, 8357, 8358, 8359, 8360, 8363, 8364, 8365, 8366, 8367, 8380, 8381, 8382, 8387, 8388, 8389, 8390, 8405, 8406, 8407, 8408, 8409, 8410, 8411, 8412, 8413, 8414, 8415, 8416, 8417, 8418, 8419, 8420, 8421, 8422, 8423, 8424, 8425, 8426, 8433, 8434, 8435, 8436, 8442, 8443, 8444, 8445, 8446, 8447, 8455, 8456, 8457, 8458, 8459, 8460, 8461, 8464, 8465, 8466, 8467, 8468, 8480, 8481, 8482, 8483, 8486, 8487, 8488, 8489, 8506, 8507, 8508, 8509, 8510, 8511, 8512, 8513, 8514, 8515, 8516, 8517, 8518, 8519, 8520, 8521, 8522, 8523, 8524, 8525, 8526, 8535, 8536, 8537, 8538, 8542, 8543, 8544, 8545, 8546, 8547, 8557, 8558, 8559, 8560, 8561, 8564, 8565, 8566, 8567, 8568, 8569, 8581, 8582, 8583, 8584, 8586, 8587, 8588, 8589, 8606, 8607, 8608, 8609, 8610, 8611, 8612, 8613, 8614, 8615, 8616, 8617, 8618, 8619, 8620, 8621, 8622, 8623, 8624, 8625, 8626, 8638, 8639, 8640, 8642, 8643, 8644, 8645, 8646, 8647, 8659, 8660, 8661, 8664, 8665, 8666, 8667, 8668, 8669, 8670, 8683, 8684, 8685, 8686, 8687, 8688, 8707, 8708, 8709, 8710, 8711, 8712, 8713, 8714, 8715, 8716, 8717, 8718, 8719, 8720, 8721, 8722, 8723, 8724, 8725, 8726, 8740, 8741, 8742, 8743, 8744, 8745, 8746, 8747, 8761, 8764, 8765, 8766, 8767, 8768, 8769, 8770, 8785, 8786, 8810, 8811, 8812, 8813, 8814, 8815, 8816, 8817, 8818, 8819, 8820, 8821, 8822, 8823, 8824, 8825, 8826, 8827, 8842, 8843, 8844, 8845, 8846, 8847, 8855, 8856, 8865, 8866, 8867, 8868, 8869, 8870, 8887, 8913, 8914, 8915, 8916, 8917, 8918, 8919, 8920, 8921, 8922, 8923, 8924, 8925, 8926, 8944, 8945, 8946, 8947, 8955, 8956, 8957, 8958, 9018, 9019, 9020, 9021, 9022, 9023, 9024, 9025, 9026, 9027, 9035, 9038, 9046, 9047, 9048, 9055, 9056, 9057, 9058, 9059, 9090, 9121, 9122, 9123, 9124, 9125, 9126, 9127, 9135, 9136, 9137, 9138, 9139, 9140, 9148, 9155, 9156, 9157, 9158, 9159, 9181, 9182, 9183, 9184, 9191, 9192, 9224, 9225, 9226, 9227, 9235, 9236, 9237, 9238, 9239, 9240, 9241, 9242, 9255, 9256, 9257, 9258, 9259, 9281, 9282, 9283, 9284, 9285, 9286, 9287, 9292, 9326, 9327, 9328, 9336, 9337, 9338, 9339, 9340, 9341, 9342, 9343, 9344, 9355, 9356, 9357, 9358, 9359, 9363, 9364, 9381, 9382, 9383, 9384, 9385, 9386, 9387, 9388, 9436, 9437, 9438, 9439, 9440, 9441, 9442, 9443, 9444, 9445, 9446, 9457, 9458, 9459, 9463, 9464, 9465, 9466, 9483, 9484, 9485, 9486, 9487, 9488, 9537, 9538, 9539, 9540, 9541, 9542, 9543, 9544, 9545, 9546, 9547, 9548, 9549, 9563, 9564, 9565, 9566, 9567, 9568, 9584, 9585, 9586, 9587, 9588, 9589, 9638, 9639, 9640, 9641, 9642, 9643, 9644, 9645, 9646, 9647, 9648, 9649, 9650, 9664, 9665, 9666, 9667, 9668, 9687, 9688, 9689, 9690, 9739, 9740, 9741, 9742, 9743, 9744, 9745, 9746, 9747, 9748, 9749, 9750, 9751, 9766, 9767, 9768, 9780, 9781, 9790, 9791, 9841, 9842, 9843, 9844, 9845, 9846, 9847, 9848, 9849, 9850, 9851, 9880, 9881, 9882, 9883, 9884, 9942, 9943, 9944, 9945, 9946, 9947, 9948, 9949, 9957, 9958, 9959, 9980, 9981, 9982, 9983, 9984, 9985, 9986, 9987, 9988, 10044, 10045, 10046, 10047, 10048, 10049, 10057, 10058, 10059, 10060, 10061, 10062, 10080, 10081, 10082, 10083, 10084, 10085, 10086, 10087, 10088, 10089, 10090, 10147, 10148, 10149, 10158, 10159, 10160, 10161, 10162, 10163, 10164, 10180, 10181, 10182, 10183, 10184, 10185, 10186, 10187, 10188, 10189, 10190, 10191, 10192]
        self.states = np.zeros(self.x*self.x).tolist()
        self.actions_size = len(self.action_space)
        self.states_size = self.x*self.y
        self.state_matrix = np.ascontiguousarray(np.arange(self.states_size).reshape(self.y, self.x), dtype=int)
        self.matrix_list = self.state_matrix.tolist()

    def reset_enviroment(self):
        self.x_position = 0
        self.y_position = 0
        self.state = 0
        self.reward = 0
        return 0, 0, False

    def update_instructions_list(self,state, state_list):
        if state not in self.obstacles_map and state!=self.x_position:
            if self.states[state]==0:
                self.states[state] = len(state_list)
            else:
                if self.states[state]> len(state_list):
                    self.states[state] = len(state_list)

    def get_env_states(self):
        return self.states

    def next_step(self, action, state):
        old_x= self.x_position
        old_y= self.y_position
        if self.x_position==self.goalx and self.y_position==self.goaly:
            done=True
            reward = 5000
            obstacle = False
            return state, reward, done, obstacle
        else:
            # up
            if action==0 and self.x_position>0:
                self.x_position -= 1
            # Move left
            elif action== 1 and self.y_position>0:
                self.y_position-= 1
            # Move down
            elif action==2 and self.x_position<self.x-1:
                self.x_position+=1
            # Move right
            elif action==3 and self.y_position<self.y-1:
                self.y_position+=1
            obstacle = False
            reward=-1
            done=False
            next_state = self.state_matrix[self.x_position][self.y_position]
            if next_state in self.obstacles_map:
                reward = -500
                obstacle = True
                #print(old_x, old_y)
                #print(self.x_position, self.y_position)
                '''                self.x_position = old_x
                                self.y_position = old_y
                                #print('novos', self.x_position, self.y_position)
                                next_state=state'''
            return next_state, reward, done, obstacle

    def get_state_index(self):
        return self.x_position, self.y_position
    
    def insert_old_state_index(self, x, y):
        self.x_old_state = x
        self.y_old_state = y

    def get_goal_index(self):
        return self.goaly, self.goaly

    def select_random_action(self):
        return np.random.choice(self.action_space)
