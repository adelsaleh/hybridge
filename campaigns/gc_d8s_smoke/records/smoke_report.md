# D8-S smoke campaign

Synchronized step time = median of steps 3..N; 'hard' replaces the warm Poisson Krylov time by the cold first solve's (reported, not ranked on). Times in ms, memory in MiB (nvidia-smi, per process).

## Stage A

| run | triangles | step | hard | Poisson RHS / solve / recon / post | transport asm / solve | it P / T (cold P) | peak | W / °C (busy) | checks |
|---|---|---|---|---|---|---|---|---|---|
| A_m128L3_p1_amgx-cheb-bsr_fused-csr_raw | 2988670 | 263 | 2030 | 32.9/44/17.1/0 | 56.4/105 | 2.5 / 16 (108) | 16176 | 189 / 61 | ok |
| A_m128L3_p2_amgx-cheb-csr_fused-csr_raw | 1496624 | 242.3 | 2077 | 29.6/22.1/12.7/0 | 59.2/105 | 1 / 19.5 (121) | 13606 | 203 / 70 | ok |
| A_m128L3_p4_fbmg-fast_fused-bsr_raw | 598613 | 273.6 | 1093 | 31.5/32.6/11.2/0 | 82.5/105 | 4 / 20.5 (127) | 12644 | 203 / 60 | ok |
| A_m128L3_p4_fbmg-standard_fused-bsr_raw | 598613 | 272.3 | 1124 | 32.9/41.6/10.8/0 | 81.8/99.8 | 3.5 / 19.5 (82) | 13814 | 226 / 60 | ok |
| A_m128L3_p5_amgx-cheb-bsr_fused-bsr_raw | 425730 | 275.2 | 1889 | 33/15.1/9.51/0 | 102/104 | 1 / 28.5 (20) | 16984 | 300 / 69 | ok |
| A_m128L3_p5_amgx-cheb-csr_fused-bsr_raw | 425730 | 274.2 | 1883 | 32.9/19.7/9.82/0 | 102/101 | 1 / 27 (20) | 18678 | 300 / 69 | ok |
| A_m128L3_p5_amgx-classical-abs-csr_fused-bsr_raw | 425730 | 332.4 | 1885 | 33/88.4/9.66/0 | 102/89.7 | 1 / 22.5 (20) | 14988 | 210 / 63 | ok |
| A_m128L3_p5_fbmg-fast_fused-bsr_raw | 425730 | 256.4 | 911.2 | 32.9/12.2/9.55/0 | 102/89.1 | 1.5 / 22 (121) | 12618 | 120 / 63 | ok |
| A_m128L3_p5_fbmg-standard_fused-bsr_raw | 425730 | 268.7 | 922.2 | 32.9/25.2/9.51/0 | 102/94.6 | 2.5 / 24.5 (79) | 12536 | 123 / 63 | ok |
| A_m128L3_p6_amgx-cheb-bsr_fused-bsr_raw | 319605 | 325.3 | 1745 | 36.1/22.8/9.16/0 | 142/105 | 2 / 32.5 (19) | 16452 | 300 / 73 | ok |
| A_m128L3_p6_amgx-cheb-csr_fused-bsr_raw | 319605 | 330.6 | 1744 | 36.2/31.3/9.02/0 | 143/102 | 2 / 31 (19) | 17976 | 296 / 70 | ok |
| A_m128L3_p6_amgx-classical-abs-csr_fused-bsr_raw | 319605 | 365 | 1739 | 36.2/82.8/9.19/0 | 142/85.7 | 1 / 23.5 (19) | 15376 | 200 / 63 | ok |
| A_m128L3_p6_fbmg-fast_fused-bsr_raw | 319605 | 302.1 | 879 | 36.5/8.87/9.31/0 | 143/95.2 | 1 / 27 (119) | 13292 | 193 / 61 | ok |
| A_m128L3_p6_fbmg-standard_fused-bsr_raw | 319605 | 320.7 | 890.3 | 36.9/20.7/9.42/0 | 143/104 | 2 / 28.5 (75) | 14458 | 163 / 61 | ok |
| A_m128L3_p7_amgx-cheb-bsr_fused-bsr_raw | 249084 | 391.7 | 1776 | 46.8/21.4/14.3/0 | 199/104 | 2 / 33.5 (20) | 16944 | 265 / 59 | ok |
| A_m128L3_p7_amgx-cheb-csr_fused-bsr_raw | 249084 | 400.3 | 1780 | 46.9/30.7/14.4/0 | 200/99.9 | 2 / 32.5 (20) | 20198 | 299 / 70 | ok |
| A_m128L3_p7_amgx-classical-abs-csr_fused-bsr_raw | 249084 | 437.5 | 1783 | 46.7/77.3/14.3/0 | 199/90.6 | 1 / 28 (20) | 16696 | 191 / 65 | ok |
| A_m128L3_p7_amgx-classical-csr_fused-bsr_raw | 249084 | 905.1 | 1390 | 48.5/524/15.6/0 | 211/87.9 | 14 / 21 (28) | 17636 | 203 / 63 | ok |
| A_m128L3_p7_fbmg-fast_fused-bsr_raw | 249084 | 385 | 882.9 | 46.8/8.41/14.4/0 | 200/105 | 1 / 35 (113) | 15628 | 195 / 66 | ok |
| A_m128L3_p7_fbmg-standard_fused-bsr_raw | 249084 | 376.6 | 865.8 | 46.7/17.5/14.3/0 | 199/89.7 | 2 / 27.5 (73) | 14520 | 185 / 63 | ok |
| A_m128L3_p8_amgx-cheb-bsr_split3-bsr_raw | 200077 | 640.1 | 1814 | 55.4/30/14.1/0 | 342/190 | 1.5 / 40.5 (18) | 27638 | 227 / 75 | ok |
| A_m128L3_p8_amgx-cheb-csr_split3-bsr_raw | 200077 | 636 | 1817 | 55.6/24.6/14.2/0 | 341/186 | 1.5 / 39.5 (18) | 27316 | 227 / 75 | ok |
| A_m128L3_p8_amgx-classical-abs-csr_split3-bsr_raw | 200077 | 648.8 | 1791 | 55.5/74/13.9/0 | 341/146 | 1 / 29 (18) | 24926 | 211 / 71 | ok |
| A_m128L3_p8_amgx-classical-csr_split3-bsr_raw | 200077 | 1023 | 1484 | 55.4/462/14.1/0 | 341/140 | 14 / 27.5 (28) | 24886 | 215 / 67 | ok |
| A_m128L3_p8_fbmg-fast_split3-bsr_raw | 200077 | 603.4 | 1264 | 55.4/16.3/14.1/0 | 341/168 | 1.5 / 35 (112) | 23006 | 209 / 69 | ok |
| A_m128L3_p8_fbmg-standard_split3-bsr_raw | 200077 | 616.4 | 1207 | 55.5/21/14.1/0 | 343/171 | 2 / 35.5 (71) | 23006 | 234 / 84 | ok |
| A_m128L3_p9_amgx-cheb-bsr_split3-bsr_raw | 164638 | 800.1 | 1917 | 86.5/26.2/13.8/0 | 499/165 | 1.5 / 39 (18) | 29594 | 218 / 76 | ok |
| A_m128L3_p9_amgx-cheb-csr_split3-bsr_raw | 164638 | 809.8 | 1940 | 87.6/24.7/14/0 | 508/163 | 1.5 / 38.5 (18) | 31196 | 212 / 72 | ok |
| A_m128L3_p9_amgx-classical-abs-csr_split3-bsr_raw | 164638 | 837.6 | 1923 | 86.2/70.5/13.7/0 | 498/158 | 1 / 38 (18) | 28392 | 204 / 69 | ok |
| A_m128L3_p9_amgx-classical-csr_split3-bsr_raw | 164638 | 1228 | 1652 | 86.8/478/14.2/0 | 503/131 | 15.5 / 29 (28) | 28360 | 205 / 71 | ok |
| A_m128L3_p9_fbmg-fast_split3-bsr_raw | 164638 | 777.8 | 1335 | 86.2/12/13.8/0 | 498/153 | 1.5 / 36 (106) | 25436 | 210 / 71 | ok |
| A_m128L3_p9_fbmg-standard_split3-bsr_raw | 164638 | 794.8 | 1295 | 86.1/19.1/13.6/0 | 498/168 | 2 / 41 (68) | 25436 | 209 / 70 | ok |

## Stage B

| run | triangles | step | hard | Poisson RHS / solve / recon / post | transport asm / solve | it P / T (cold P) | peak | W / °C (busy) | checks |
|---|---|---|---|---|---|---|---|---|---|
| B_m128L3_p6_fbmg-fast-lu_fused-bsr_raw | 319605 | 342.2 | 916.4 | 50.7/8.89/36.4/0 | 141/93.8 | 1 / 27 (119) | 11814 | 213 / 61 | ok |
| B_m128L3_p8_amgx-cheb-bsr-modal_split3-bsr_raw | 200077 | 865.8 | 6.281e+04 | 55.5/291/14.2/0 | 342/155 | 18.5 / 31.5 (1067) | 26792 | 234 / 84 | ok |
| B_m128L3_p9_amgx-cheb-bsr-bs128_split3-bsr_raw | 164638 | 798.5 | 1915 | 86.2/26.2/14.1/0 | 499/162 | 1.5 / 39 (18) | 28966 | 223 / 79 | ok |
| B_m128L3_p9_amgx-cheb-bsr-bs64_split3-bsr_raw | 164638 | 799 | 1916 | 87.2/26.2/13.7/0 | 499/163 | 1.5 / 39 (18) | 28462 | 220 / 78 | ok |
| B_m128L3_p9_amgx-cheb-bsr-lu_split3-bsr_raw | 164638 | 893 | 2011 | 99.7/26.2/93.9/0 | 499/164 | 1.5 / 39.5 (18) | 26388 | 219 / 77 | ok |

## Stage C

| run | triangles | step | hard | Poisson RHS / solve / recon / post | transport asm / solve | it P / T (cold P) | peak | W / °C (busy) | checks |
|---|---|---|---|---|---|---|---|---|---|
| C_m128L3_p5_fbmg-fast_fused-bsr-bj_raw | 425730 | 251.1 | 906.7 | 32.8/12.2/9.52/0 | 102/85.6 | 1.5 / 8 (121) | 13584 | 160 / 63 | ok |
| C_m128L3_p5_fbmg-standard_fused-bsr-bj_raw | 425730 | 265.2 | 919.6 | 33/25.3/9.5/0 | 102/87.9 | 2.5 / 8 (79) | 12618 | 168 / 63 | ok |
| C_m128L3_p6_fbmg-fast_fused-bsr-bj_raw | 319605 | 282.7 | 857 | 36.3/8.9/9.02/0 | 141/76.1 | 1 / 7.5 (119) | 13292 | 229 / 64 | ok |
| C_m128L3_p6_fbmg-fast_split3-bsr_raw | 319605 | 291.5 | 870.9 | 36.2/8.89/9.11/0 | 134/93.5 | 1 / 27 (119) | 15484 | 211 / 66 | ok |
| C_m128L3_p6_fbmg-standard_fused-bsr-bj_raw | 319605 | 292.3 | 853.2 | 36/19.2/9.18/0 | 141/78.3 | 2 / 8 (75) | 13292 | 190 / 61 | ok |
| C_m128L3_p6_fbmg-standard_split3-bsr_raw | 319605 | 307.2 | 868.6 | 36.2/19.2/9.2/0 | 134/97.3 | 2 / 28.5 (75) | 15484 | 216 / 64 | ok |
| C_m128L3_p7_amgx-cheb-bsr_fused-bsr-bj_raw | 249084 | 360.8 | 1748 | 46.7/21.4/14.2/0 | 200/72.7 | 2 / 8 (20) | 16944 | 300 / 68 | ok |
| C_m128L3_p7_amgx-cheb-bsr_split3-bsr-bj_raw | 249084 | 348 | 1735 | 46.9/21.4/14.3/0 | 185/72.9 | 2 / 8 (20) | 19654 | 230 / 69 | ok |
| C_m128L3_p7_amgx-cheb-bsr_split3-bsr_raw | 249084 | 376.8 | 1764 | 46.7/21.4/14.3/0 | 185/102 | 2 / 33.5 (20) | 19654 | 253 / 69 | ok |
| C_m128L3_p7_fbmg-standard_fused-bsr-bj_raw | 249084 | 361 | 850.3 | 46.8/17.7/14.5/0 | 199/74.2 | 2 / 8 (73) | 14520 | 188 / 62 | ok |
| C_m128L3_p7_fbmg-standard_split3-bsr-bj_raw | 249084 | 347.5 | 837.7 | 46.9/17.6/14.5/0 | 185/74.2 | 2 / 8 (73) | 17156 | 191 / 65 | ok |
| C_m128L3_p7_fbmg-standard_split3-bsr_raw | 249084 | 366.4 | 860 | 46.9/17.6/14.5/0 | 186/91 | 2 / 27.5 (73) | 17156 | 202 / 73 | ok |
| C_m128L3_p8_amgx-cheb-bsr_fused-bsr_raw | 200077 | 757.4 | 1931 | 55.4/30/14/0 | 460/189 | 1.5 / 40.5 (18) | 23454 | 273 / 72 | ok |
| C_m128L3_p8_amgx-cheb-bsr_split3-bsr-bj_raw | 200077 | 572.5 | 1746 | 55.3/30/13.9/0 | 342/121 | 1.5 / 8.5 (18) | 27566 | 228 / 76 | ok |
| C_m128L3_p8_amgx-cheb-bsr_split3-bsr-l1_raw | 200077 | 865.5 | 2039 | 55.3/30.1/14.1/0 | 342/415 | 1.5 / 95.5 (18) | 28062 | 237 / 75 | ok |
| C_m128L3_p8_amgx-cheb-csr_fused-bsr_raw | 200077 | 754.4 | 1937 | 55.5/24.6/13.9/0 | 460/187 | 1.5 / 39.5 (18) | 24224 | 211 / 71 | ok |
| C_m128L3_p8_amgx-cheb-csr_split3-bsr-bj_raw | 200077 | 566.9 | 1749 | 55.3/24.6/14/0 | 341/125 | 1.5 / 9 (18) | 28336 | 228 / 74 | ok |
| C_m128L3_p8_amgx-cheb-csr_split3-bsr-l1_raw | 200077 | 836.6 | 2018 | 55.4/24.7/14/0 | 341/392 | 1.5 / 90 (18) | 27316 | 237 / 74 | ok |
| C_m128L3_p8_fbmg-standard_fused-bsr_raw | 200077 | 733.1 | 1308 | 55.7/20.9/14.3/0 | 459/170 | 2 / 35.5 (71) | 20982 | 184 / 64 | ok |
| C_m128L3_p8_fbmg-standard_split3-bsr-bj_raw | 200077 | 558.3 | 1132 | 56/21/14.1/0 | 341/116 | 2 / 8 (71) | 24350 | 209 / 68 | ok |
| C_m128L3_p8_fbmg-standard_split3-bsr-l1_raw | 200077 | 803.9 | 1380 | 55.7/20.9/14.1/0 | 341/357 | 2 / 81 (71) | 23006 | 220 / 70 | ok |
| C_m128L3_p9_amgx-cheb-bsr-bs128_fused-bsr_raw | 164638 | 1220 | 2337 | 86.2/26.3/14.2/0 | 922/162 | 1.5 / 39 (18) | 25698 | 187 / 73 | ok |
| C_m128L3_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_raw | 164638 | 744.7 | 1861 | 86.3/26.2/14.1/0 | 500/111 | 1.5 / 9 (18) | 29498 | 219 / 76 | ok |
| C_m128L3_p9_fbmg-standard_fused-bsr_raw | 164638 | 1219 | 1719 | 86.2/19.2/13.9/0 | 920/169 | 2 / 41 (68) | 21636 | 167 / 67 | ok |
| C_m128L3_p9_fbmg-standard_split3-bsr-bj_raw | 164638 | 753.4 | 1252 | 88.2/20.5/15.5/0 | 500/114 | 2 / 8 (68) | 25436 | 201 / 70 | ok |

## Stage D

| run | triangles | step | hard | Poisson RHS / solve / recon / post | transport asm / solve | it P / T (cold P) | peak | W / °C (busy) | checks |
|---|---|---|---|---|---|---|---|---|---|
| D_m128L3_p5_fbmg-standard_fused-bsr-bj_rawq6 | 425730 | 289.4 | 994.9 | 47.2/25.5/11.4/0 | 107/87.2 | 2 / 8 (71) | 16284 | 169 / 64 | ok |
| D_m128L3_p5_fbmg-standard_fused-bsr-bj_v2 | 425730 | 276.8 | 935.5 | 33/25.3/9.77/3.88 | 102/92.5 | 2.5 / 9 (79) | 17950 | 187 / 62 | ok |
| D_m128L3_p6_amgx-cheb-bsr_fused-bsr-bj_rawq7 | 319605 | 333.8 | 2129 | 59.6/27.5/18.2/0 | 143/76.1 | 2 / 7.5 (21) | 21680 | 282 / 69 | ok |
| D_m128L3_p6_fbmg-standard_fused-bsr-bj_rawq7 | 319605 | 330.6 | 901.1 | 59.3/22.3/17.6/0 | 142/78.1 | 2 / 8 (67) | 17224 | 171 / 63 | ok |
| D_m128L3_p6_fbmg-standard_fused-bsr-bj_v2 | 319605 | 311 | 871.6 | 36.5/19.6/9.43/4.81 | 141/84.8 | 2 / 9 (75) | 19210 | 129 / 61 | ok |
| D_m128L3_p7_amgx-cheb-bsr_fused-bsr-bj_v2 | 249084 | 378.4 | 1765 | 47.3/21.6/14.5/5.89 | 200/76.8 | 2 / 9 (20) | 24190 | 240 / 67 | ok |
| D_m128L3_p7_amgx-cheb-csr_fused-bsr-bj_rawq8 | 249084 | 395.4 | 1886 | 70.4/23.1/16.9/0 | 201/73.3 | 1 / 8 (19) | 27710 | 300 / 72 | ok |
| D_m128L3_p7_fbmg-standard_split3-bsr-bj_rawq8 | 249084 | 384.7 | 1055 | 70.1/25.9/16.9/0 | 186/73.9 | 2 / 8 (67) | 23428 | 186 / 68 | ok |
| D_m128L3_p7_fbmg-standard_split3-bsr-bj_v2 | 249084 | 359.9 | 848.7 | 46.8/17.5/14.3/5.9 | 185/78.4 | 2 / 9 (73) | 23908 | 199 / 65 | ok |
| D_m128L3_p8_amgx-cheb-bsr-bs128_split3-bsr-bj_rawq9 | 200077 | 655.1 | 2029 | 104/31.7/16.9/0 | 368/119 | 1.5 / 8 (19) | 31312 | 232 / 78 | ok |
| D_m128L3_p8_amgx-cheb-csr_split3-bsr-bj_v2 | 200077 | 583.4 | 1766 | 55.8/25.1/14.5/7.09 | 342/126 | 1.5 / 9 (18) | 35968 | 224 / 73 | ok |
| D_m128L3_p8_fbmg-standard_split3-bsr-bj_rawq9 | 200077 | 637.3 | 1211 | 104/22.9/16.2/0 | 367/117 | 2 / 8 (65) | 27936 | 207 / 69 | ok |
| D_m128L3_p8_fbmg-standard_split3-bsr-bj_v2 | 200077 | 581 | 1155 | 55.4/20.9/14.1/7.1 | 341/129 | 2 / 9 (71) | 31488 | 208 / 70 | ok |
| D_m128L3_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_v2 | 164638 | 764.6 | 1882 | 87/26.7/14.6/8.46 | 500/114 | 1.5 / 9 (18) | 38086 | 216 / 75 | ok |
| D_m128L3_p9_fbmg-standard_split3-bsr-bj_v2 | 164638 | 752.8 | 1253 | 86.2/19.1/14/8.47 | 498/112 | 2 / 9 (68) | 35100 | 204 / 70 | ok |

## Stage E

| run | triangles | step | hard | Poisson RHS / solve / recon / post | transport asm / solve | it P / T (cold P) | peak | W / °C (busy) | checks |
|---|---|---|---|---|---|---|---|---|---|
| E_m128L3_p5_fbmg-standard_fused-bsr-bj_raw_vq8 | 425730 | 277.1 | 932.8 | 33.1/25.4/9.8/0 | 113/88.1 | 2.5 / 8 (79) | 15810 | 198 / 63 | ok |
| E_m128L3_p6_fbmg-standard_fused-bsr-bj_raw_vq10 | 319605 | 315.6 | 876.5 | 36.6/19.5/9.3/0 | 159/79.9 | 2 / 8 (75) | 15128 | 177 / 63 | ok |
| E_m128L3_p7_amgx-cheb-bsr_fused-bsr-bj_raw_vq11 | 249084 | 396 | 1785 | 47.2/21.6/14.5/0 | 232/73.2 | 2 / 8 (20) | 19658 | 256 / 67 | ok |
| E_m128L3_p7_fbmg-standard_split3-bsr-bj_raw_vq11 | 249084 | 388 | 878.6 | 46.7/17.5/14.5/0 | 225/73.3 | 2 / 8 (73) | 18966 | 205 / 66 | ok |
| E_m128L3_p8_amgx-cheb-csr_split3-bsr-bj_raw_vq13 | 200077 | 568.7 | 1750 | 55.6/24.8/14.1/0 | 342/125 | 1.5 / 9 (18) | 26426 | 227 / 75 | ok |
| E_m128L3_p8_fbmg-standard_split3-bsr-bj_raw_vq13 | 200077 | 553.9 | 1128 | 55.4/20.9/14.1/0 | 341/116 | 2 / 8 (71) | 20718 | 218 / 68 | ok |
| E_m128L3_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_raw_vq14 | 164638 | 748.4 | 1868 | 86.7/26.4/14.6/0 | 500/112 | 1.5 / 9 (18) | 25558 | 214 / 75 | ok |
| E_m128L3_p9_fbmg-standard_split3-bsr-bj_raw_vq14 | 164638 | 733.3 | 1233 | 86.2/19.1/13.7/0 | 498/106 | 2 / 8 (68) | 21872 | 206 / 69 | ok |

## Stage F

| run | triangles | step | hard | Poisson RHS / solve / recon / post | transport asm / solve | it P / T (cold P) | peak | W / °C (busy) | checks |
|---|---|---|---|---|---|---|---|---|---|
| F_m128L4_p5_fbmg-standard_fused-bsr-bj_raw | 1283577 | 839.3 | 2392 | 103/113/24.4/0 | 307/266 | 4 / 8 (64) | 39646 | 219 / 64 | ok |
| F_m128L4_p8_amgx-cheb-csr_split3-bsr-bj_raw | 598613 | – | – | –/–/–/– | –/– | – / – (–) | 45936 | 300 / 85 | oom; no summary row |
| F_m128L4_p8_amgx-cheb-csr_split3-bsr-bj_v2 | 598613 | – | – | –/–/–/– | –/– | – / – (–) | 44596 | 300 / 80 | oom; no summary row |

## Stage G

| run | triangles | step | hard | Poisson RHS / solve / recon / post | transport asm / solve | it P / T (cold P) | peak | W / °C (busy) | checks |
|---|---|---|---|---|---|---|---|---|---|
| G_zgL4_p5_fbmg-standard_fused-bsr-bj_raw | 423907 | 322 | 611.5 | 35.2/21.1/9.76/0 | 102/144 | 2 / 17 (36) | 13990 | 199 / 63 | ok |
| G_zgL4_p6_fbmg-standard_fused-bsr-bj_raw | 315929 | 347.2 | 597.1 | 37.9/26.5/9.26/0 | 141/123 | 3 / 16.5 (36) | 12754 | 187 / 62 | ok |
| G_zgL4_p7_amgx-cheb-bsr_fused-bsr-bj_raw | 250052 | 403.4 | 617.6 | 48.9/21.8/14.5/0 | 202/105 | 2 / 15 (26) | 15094 | 167 / 64 | ok |
| G_zgL4_p7_fbmg-standard_split3-bsr-bj_raw | 250052 | 388.4 | 618.9 | 49/21/14.5/0 | 186/107 | 2.5 / 15 (36) | 16850 | 192 / 65 | ok |
| G_zgL4_p8_amgx-cheb-csr_split3-bsr-bj_raw | 200004 | 633.4 | 948.6 | 57.2/36.1/14.1/0 | 341/176 | 2.5 / 14 (29) | 24754 | 211 / 68 | ok |
| G_zgL4_p8_fbmg-standard_split3-bsr-bj_raw | 200004 | 626.5 | 892.7 | 56.9/29/13.9/0 | 341/176 | 3 / 14 (35) | 24772 | 208 / 69 | ok |
| G_zgL4_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_raw | 163787 | 784.9 | 1113 | 88.2/33.2/14.4/0 | 496/143 | 2 / 13 (24) | 26110 | 200 / 71 | ok |
| G_zgL4_p9_fbmg-standard_split3-bsr-bj_raw | 163787 | 783.4 | 1030 | 88.2/26.3/13.7/0 | 496/146 | 3 / 13 (36) | 24304 | 200 / 70 | ok |

## Stage H

| run | triangles | step | hard | Poisson RHS / solve / recon / post | transport asm / solve | it P / T (cold P) | peak | W / °C (busy) | checks |
|---|---|---|---|---|---|---|---|---|---|
| H_m128L2_p5_fbmg-fast_fused-bsr-bj_raw_eps0p1 | 144708 | 126.4 | 433.1 | 13.1/34.5/5.3/0 | 35.5/32 | 15 / 8 (155) | 5218 | 233 / 85 | ok |
| H_m128L2_p5_fbmg-standard_fused-bsr-bj_raw_eps0p1 | 144708 | 120.6 | 412.7 | 13/30.4/5.12/0 | 35.4/31.1 | 9 / 8 (100) | 5218 | 242 / 86 | ok |
| H_m128L2_p6_fbmg-fast_fused-bsr-bj_raw_eps0p1 | 107116 | 136.4 | 419.7 | 14.1/34.2/5.11/0 | 48.4/29.5 | 16 / 9 (161) | 5350 | 225 / 85 | ok |
| H_m128L2_p6_fbmg-standard_fused-bsr-bj_raw_eps0p1 | 107116 | 131.9 | 398.8 | 14.1/30.4/5.11/0 | 48.2/29.6 | 10 / 9 (101) | 5378 | 229 / 85 | ok |
| H_m128L2_p7_amgx-cheb-bsr_fused-bsr-bj_raw_eps0p1 | 83591 | 270.7 | 678.6 | 17.9/144/6.96/0 | 68.2/27.6 | 41 / 9 (17) | 6902 | 248 / 87 | ok |
| H_m128L2_p7_amgx-cheb-csr_fused-bsr-bj_raw_eps0p1 | 83591 | 360.1 | 675.1 | 17.7/237/6.64/0 | 67.8/26.9 | 39 / 9 (17) | 7446 | 241 / 87 | ok |
| H_m128L2_p7_fbmg-fast_fused-bsr_raw_eps0p1 | 83591 | 156 | 385.2 | 17.7/27.5/6.72/0 | 67.8/31.9 | 14 / 26 (138) | 5782 | 212 / 84 | ok |
| H_m128L2_p7_fbmg-standard_fused-bsr_raw_eps0p1 | 83591 | 151.7 | 365.8 | 17.6/25.3/6.65/0 | 67.5/31.6 | 9 / 25 (89) | 5650 | 215 / 83 | ok |
| H_m128L2_p8_amgx-cheb-bsr_split3-bsr-bj_raw_eps0p1 | 67114 | 395 | 733.2 | 20.6/201/6.85/0 | 116/44.5 | 33 / 9 (17) | 9836 | 252 / 88 | ok |
| H_m128L2_p8_amgx-cheb-csr_split3-bsr-bj_raw_eps0p1 | 67114 | 389.3 | 734.1 | 20.6/196/6.79/0 | 116/44.4 | 33 / 9 (17) | 10558 | 242 / 87 | ok |
| H_m128L2_p8_fbmg-fast_split3-bsr_raw_eps0p1 | 67114 | 235.1 | 541.7 | 20.4/35.1/6.57/0 | 116/52.4 | 14 / 29 (143) | 8878 | 235 / 86 | ok |
| H_m128L2_p8_fbmg-standard_split3-bsr_raw_eps0p1 | 67114 | 230.9 | 487.2 | 20.6/29.9/6.72/0 | 116/53.1 | 9 / 29 (90) | 8878 | 234 / 86 | ok |
| H_m128L2_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_raw_eps0p1 | 55099 | 398.1 | 2730 | 30.8/149/6.53/0 | 169/38.7 | 29 / 9 (489) | 9860 | 244 / 88 | ok |
| H_m128L2_p9_amgx-cheb-csr_split3-bsr-bj_raw_eps0p1 | 55099 | 418.3 | 746.1 | 30.9/169/6.42/0 | 169/39.5 | 29 / 9 (16) | 11164 | 236 / 86 | ok |
| H_m128L2_p9_fbmg-fast_split3-bsr_raw_eps0p1 | 55099 | 291.1 | 538.6 | 30.9/31.2/6.46/0 | 169/49.5 | 14 / 30 (131) | 9558 | 225 / 85 | ok |
| H_m128L2_p9_fbmg-standard_split3-bsr_raw_eps0p1 | 55099 | 285.8 | 506.7 | 30.8/24.5/6.43/0 | 168/49.7 | 8 / 31 (85) | 9558 | 225 / 85 | ok |

## Selected presets

| p | level | Poisson | transport | step | hard | saturated-phase step (L2) |
|---|---|---|---|---|---|---|
| 1 | L3 | amgx-cheb-bsr | fused+csr | 263 | 2030 | – |
| 2 | L3 | amgx-cheb-csr | fused+csr | 242.3 | 2077 | – |
| 4 | L3 | fbmg-fast | fused+bsr | 273.6 | 1093 | – |
| 5 | L3 | fbmg-standard | fused+bsr+bj | 265.2 | 919.6 | 128.2 |
| 6 | L3 | fbmg-standard | fused+bsr+bj | 292.3 | 853.2 | 135.7 |
| 7 | L3 | fbmg-standard | split3+bsr+bj | 347.5 | 837.7 | 157.6 |
| 8 | L3 | fbmg-standard | split3+bsr+bj | 558.3 | 1132 | 238.1 |
| 9 | L3 | fbmg-standard | split3+bsr+bj | 753.4 | 1252 | 290.1 |

## Saturated phase (stage H: seed 0.1, L2, medians over t >= 15)

| run | step | Poisson Krylov | it P / T | steps | checks |
|---|---|---|---|---|---|
| H_m128L2_p5_fbmg-fast_fused-bsr-bj_raw_eps0p1 | 133.5 | 40.8 | 18 / 9 | 41 | ok |
| H_m128L2_p5_fbmg-standard_fused-bsr-bj_raw_eps0p1 | 128.2 | 39.8 | 12 / 9 | 41 | ok |
| H_m128L2_p6_fbmg-fast_fused-bsr-bj_raw_eps0p1 | 141.4 | 38.3 | 19 / 9 | 41 | ok |
| H_m128L2_p6_fbmg-standard_fused-bsr-bj_raw_eps0p1 | 135.7 | 30.3 | 10 / 9 | 41 | ok |
| H_m128L2_p7_amgx-cheb-bsr_fused-bsr-bj_raw_eps0p1 | 276.4 | 148 | 43 / 9 | 41 | ok |
| H_m128L2_p7_amgx-cheb-csr_fused-bsr-bj_raw_eps0p1 | 372.6 | 247 | 41 / 9 | 41 | ok |
| H_m128L2_p7_fbmg-fast_fused-bsr_raw_eps0p1 | 161.5 | 32.4 | 17 / 28 | 41 | ok |
| H_m128L2_p7_fbmg-standard_fused-bsr_raw_eps0p1 | 157.6 | 27.6 | 10 / 28 | 41 | ok |
| H_m128L2_p8_amgx-cheb-bsr_split3-bsr-bj_raw_eps0p1 | 407 | 194 | 32 / 9 | 41 | ok |
| H_m128L2_p8_amgx-cheb-csr_split3-bsr-bj_raw_eps0p1 | 393.9 | 199 | 34 / 9 | 41 | ok |
| H_m128L2_p8_fbmg-fast_split3-bsr_raw_eps0p1 | 244.1 | 41.7 | 17 / 31 | 41 | ok |
| H_m128L2_p8_fbmg-standard_split3-bsr_raw_eps0p1 | 238.1 | 32.9 | 10 / 31 | 41 | ok |
| H_m128L2_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_raw_eps0p1 | 399.3 | 147 | 29 / 9 | 41 | ok |
| H_m128L2_p9_amgx-cheb-csr_split3-bsr-bj_raw_eps0p1 | 413.9 | 161 | 28 / 9 | 41 | ok |
| H_m128L2_p9_fbmg-fast_split3-bsr_raw_eps0p1 | 296.3 | 35.1 | 16 / 32 | 41 | ok |
| H_m128L2_p9_fbmg-standard_split3-bsr_raw_eps0p1 | 290.1 | 29.6 | 10 / 32 | 41 | ok |

## Drift variants (selected presets, synchronized step time, ms)

| p | raw q=p | raw q=p+1 | V2 | V2 postprocess |
|---|---|---|---|---|
| 1 | 263 | – | – | – |
| 2 | 242.3 | – | – | – |
| 3 | – | – | – | – |
| 4 | 273.6 | – | – | – |
| 5 | 265.2 | 289.4 | 276.8 | 3.88 |
| 6 | 292.3 | 333.8 | 311 | 4.81 |
| 7 | 347.5 | 395.4 | 378.4 | 5.89 |
| 8 | 558.3 | 655.1 | 583.4 | 7.09 |
| 9 | 753.4 | – | 764.6 | 8.46 |

## Consistency of runs solving the same discrete problem

83 pairs; largest relative difference of the final density coefficients: 7.44e-07.

- H_m128L2_p9_fbmg-fast_split3-bsr_raw_eps0p1 vs H_m128L2_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_raw_eps0p1: 7.44e-07
- H_m128L2_p9_amgx-cheb-csr_split3-bsr-bj_raw_eps0p1 vs H_m128L2_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_raw_eps0p1: 6.8e-07
- H_m128L2_p8_fbmg-fast_split3-bsr_raw_eps0p1 vs H_m128L2_p8_amgx-cheb-bsr_split3-bsr-bj_raw_eps0p1: 6.49e-07
- H_m128L2_p7_fbmg-fast_fused-bsr_raw_eps0p1 vs H_m128L2_p7_amgx-cheb-bsr_fused-bsr-bj_raw_eps0p1: 5.97e-07
- H_m128L2_p9_fbmg-standard_split3-bsr_raw_eps0p1 vs H_m128L2_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_raw_eps0p1: 5.86e-07
- H_m128L2_p8_amgx-cheb-csr_split3-bsr-bj_raw_eps0p1 vs H_m128L2_p8_amgx-cheb-bsr_split3-bsr-bj_raw_eps0p1: 5.77e-07
- H_m128L2_p8_fbmg-standard_split3-bsr_raw_eps0p1 vs H_m128L2_p8_amgx-cheb-bsr_split3-bsr-bj_raw_eps0p1: 5.45e-07
- H_m128L2_p7_amgx-cheb-csr_fused-bsr-bj_raw_eps0p1 vs H_m128L2_p7_amgx-cheb-bsr_fused-bsr-bj_raw_eps0p1: 5.34e-07
- H_m128L2_p7_fbmg-standard_fused-bsr_raw_eps0p1 vs H_m128L2_p7_amgx-cheb-bsr_fused-bsr-bj_raw_eps0p1: 5.18e-07
- B_m128L3_p8_amgx-cheb-bsr-modal_split3-bsr_raw vs A_m128L3_p8_amgx-cheb-bsr_split3-bsr_raw: 4.94e-07

## Volume quadrature (stage E)

- p = 5: 64 points, step 277.1 ms against 265.2 ms; density difference 1.39e-08
- p = 6: 100 points, step 315.6 ms against 292.3 ms; density difference 3.97e-09
- p = 7: 121 points, step 396 ms against 360.8 ms; density difference 2.33e-09
- p = 7: 121 points, step 388 ms against 347.5 ms; density difference 2.27e-09
- p = 8: 169 points, step 568.7 ms against 566.9 ms; density difference 6.57e-10
- p = 8: 169 points, step 553.9 ms against 558.3 ms; density difference 6.57e-10
- p = 9: 196 points, step 748.4 ms against 744.7 ms; density difference 5.81e-10
- p = 9: 196 points, step 733.3 ms against 753.4 ms; density difference 5.82e-10

## Tooling

- contaminated attempts (requeued): [('H_m128L2_p7_amgx-cheb-bsr_fused-bsr-bj_raw_eps0p1', [4124450]), ('G_zgL4_p7_amgx-cheb-bsr_fused-bsr-bj_raw', [9366]), ('G_zgL4_p9_amgx-cheb-bsr-bs128_split3-bsr-bj_raw', [13136]), ('H_m128L2_p9_fbmg-fast_split3-bsr_raw_eps0p1', [87785])]
- runs not valid: [('F_m128L4_p8_amgx-cheb-csr_split3-bsr-bj_raw', ['oom', 'no summary row']), ('F_m128L4_p8_amgx-cheb-csr_split3-bsr-bj_v2', ['oom', 'no summary row'])]
