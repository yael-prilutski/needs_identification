from os.path import join

BASE_PATH = r'\\isi.storwis.weizmann.ac.il\Labs\livneh\2p_data'
RESULTS_PATH = r'\\isi.storwis.weizmann.ac.il\Labs\livneh\yaelpri\manuscript\conflicting_needs\figures'
MANIFOLD_PATH = r'\\isi.storwis.weizmann.ac.il\Labs\livneh\yaelpri\results_summary\clustering\data\ITIs'


CUES_REGULAR = {
    'PELLET_SIGNAL': 1,
    'FLUID_SIGNAL': 2,
    'NEUTRAL_SIGNAL': 5,
    'OMISSION_FLUID_SIGNAL': 9,
    'OMISSION_PELLET_SIGNAL': 8,
    'SURPRISE_FLUID_SIGNAL': 10,
    'SURPRISE_PELLET_SIGNAL': 11,
    'FLUID_INSTEAD_OF_PELLET_SIGNAL': 12,
    'PELLET_INSTEAD_OF_FLUID_SIGNAL': 13
}

# Mice paths
MOUSE1 = {'path': join(BASE_PATH, 'YP02_01'),
          'n_cells': {'week1': 392},
          'location': {'bregma': -400, 'height': -3360, 'rotate': 308},
          'week1': '202206_12_16',
          'week1_days': ('20220612', '20220613', '20220614', '20220615a', '20220615b', '20220616'),
          'week2': '202206_19_23',
          'week2_days': ('20220619', '20220620', '20220621', '20220622a', '20220622b', '20220623'),
          'cues': CUES_REGULAR
          }

MOUSE2 = {'path': join(BASE_PATH, 'YP02_02'),
          'n_cells': {'week1': 384},
          'location': {'bregma': -1250, 'height': -3600, 'rotate': 41},
          'week1': '202206_12_16',
          'week1_days': ('20220612', '20220613', '20220614', '20220615a', '20220615b', '20220616'),
          'week2': '202206_19_23',
          'week2_days': ('20220619', '20220620', '20220621', '20220622a', '20220622b', '20220623'),
          'cues': CUES_REGULAR
          }

MOUSE3 = {'path': join(BASE_PATH, 'YP02_03'),
          'n_cells': {'week1': 722},
          'location': {'bregma': -600, 'height': -2900, 'rotate': 306},
          'week1': '202206_12_16',
          'week1_days': ('20220612', '20220613', '20220614', '20220615a', '20220615b', '20220616'),
          'week2': '202206_19_23',
          'week2_days': ('20220619', '20220620', '20220621', '20220622a', '20220622b', '20220623'),
          'cues': CUES_REGULAR
          }

MOUSE_SS3 = {'path': join(BASE_PATH, 'SS3'),
             # 'water1_week': '202211_07_09',
             # 'water1_days': ('20221107', '20221108', '20221109'),
             'water2_week': '202211_14_16',
             'water2_days': ('20221114', '20221115', '20221116'),
             # 'water3_week': '202211_20_22',
             # 'water3_days': ('20221120', '20221121', '20221122')
             }

MOUSE_SS4 = {'path': join(BASE_PATH, 'SS4'),
             'n_cells': {'week1': 222, 'predictions': 374},
             'location': {'bregma': 1005, 'height': -3800, 'rotate': 19},
             'water2_week': '202211_14_16',
             'water2_days': ('20221114', '20221115', '20221116'),
             'week1': '202301_01_05',
             'week1_days': ('20230101', '20230102', '20230103', '20230104a', '20230104b', '20230105'),
             'week2': '202301_24_29',
             'week2_days': ('20230129', '20230124', '20230125', '20230126a', '20230126b', '20230127'),
             'omission_week': '202301_30_03',
             'omission_days': ('20230130', '20230131', '20230201a', '20230201b', '20230203')
             }

MOUSE_SS6 = {'path': join(BASE_PATH, 'SS6_Yael'),
             'n_cells': {'opto_water': 363},
             'opto_water_week': '202401_14_16',
             'opto_water_days': ('20240114', '20240115', '20240116'),
             'opto_test_week': '202401_17_18',
             'opto_test_days': ('20240117', '20240118')
             }

MOUSE_SS37 = {'path': join(BASE_PATH, 'SS37_Yael'),
              'n_cells': {'opto_water': 301},
              'opto_water_week': '202408_07_09',
              'opto_water_days': ('20240807', '20240808', '20240809'),
              'opto_test_week': '202412_27_28',
              'opto_test_days': ('20241227', '20231228')
              }

MOUSE_SS38 = {'path': join(BASE_PATH, 'SS38_Yael'),
              'n_cells': {'opto_water': 539},
              'opto_water_week': '202408_07_09',
              'opto_water_days': ('20240807', '20240808', '20240809'),
              'opto_test_week': '202412_27_28',
              'opto_test_days': ('20241227', '20231228')
              }

MOUSE_YP64 = {'path': join(BASE_PATH, 'YP64'),
              'n_cells': {'week1': 112, 'opto_water': 369, 'predictions': 256},
              'location': {'bregma': -560, 'height': -3600, 'rotate': 24},
              'week2': '202304_23_27',
              'week2_days': ('20230423', '20230424', '20230425', '20230426a', '20230426b', '20230427'),
              'week1': '202305_07_11',
              'week1_days': ('20230507', '20230508', '20230509', '20230510a', '20230510b', '20230511'),
              'omission_week': '202305_23_27',
              'omission_days': ('20230523', '20230524', '20230525a', '20230525b', '20230527'),
              'opto_mix_week': '202308_21_22',
              'opto_mix_days': ('20230821', '20230822'),
              'opto_water_week': '202312_19_21',
              'opto_water_days': ('20231219', '20231220', '20231221'),
              'opto_test_week': '202312_27_28',
              'opto_test_days': ('20231227', '20231228')
              }

MOUSE_YP79 = {'path': join(BASE_PATH, 'YP79'),
              'n_cells': {'week1': 271, 'opto_water': 313, 'opto_food': 273, 'predictions': 295},
              'location': {'bregma': 320, 'height': -3400, 'rotate': 318},
              'week1': '202308_26_30',
              'week1_days': ('20230826', '20230827', '20230828', '20230829a', '20230829b', '20230830'),
              'opto_water_week': '202309_04_06',
              'opto_water_days': ('20230904', '20230905', '20230906'),
              'opto_food_week': '202309_10_14',
              'opto_food_days': ('20230910a', '20230910b', '20230912a', '20230912b', '20230914a', '20230914b'),
              'omission_week': '202309_26_28',
              'omission_days': ('20230926', '20230927', '20230928a', '20230928b'),
              'extinction_week': '202310_01_02',
              'extinction_days': ('20231001', '20231002'),
              'opto_test_days': ('20231105', '20231106'),
              'opto_test_week': '202311_05_06',
              'opto_test_thirsty_days': ('20240121', '20240122'),
              'opto_test_thirsty_week': '202401_21_22'
              }

MOUSE_YP81 = {'path': join(BASE_PATH, 'YP81'),
              'n_cells': {'opto_water': 100},
              'location': {'bregma': -700, 'height': -4000, 'rotate': 331},
              'opto_water_week': '202312_19_21',
              'opto_water_days': ('20231219', '20231220', '20231221'),
              'opto_test_week': '202312_27_28',
              'opto_test_days': ('20231227', '20231228')
              }

MOUSE_YP82 = {'path': join(BASE_PATH, 'YP82'),
              'n_cells': {'week1': 422, 'opto_water': 496, 'opto_food': 397, 'predictions': 449},
              'location': {'bregma': 360, 'height': -3700, 'rotate': 324},
              'week1': '202311_05_09',
              'week1_days': ('20231105', '20231106', '20231107', '20231108a', '20231108b', '20231109'),
              'opto_water_week_bad': '202311_13_15',
              'opto_water_days_bad': ('20231113', '20231114', '20231115'),
              'opto_food_week': '202311_19_23',
              'opto_food_days': ('20231119a', '20231119b', '20231121a', '20231121b', '20231123a', '20231123b'),
              'omission_week': '202312_03_05',
              'omission_days': ('20231203', '20231204', '20231205a', '20231205b'),
              'extinction_week': '202312_10_11',
              'extinction_days': ('20231210', '20231211'),
              'opto_test_days': ('20231213', '20231214'),
              'opto_test_week': '202312_13_14',
              'opto_water_week': '202312_25_27',
              'opto_water_days': ('20231225', '20231226', '20231227')
              }

MOUSE_YP83 = {'path': join(BASE_PATH, 'YP83'),
              'n_cells': {'week1': 120, 'opto_water': 165, 'opto_food': 127, 'predictions': 130},
              'location': {'bregma': -500, 'height': -2900, 'rotate': 316},
              'week1': '202311_05_09',
              'week1_days': ('20231105', '20231106', '20231107', '20231108a', '20231108b', '20231109'),
              'opto_water_week': '202311_14_16',
              'opto_water_days': ('20231114', '20231115', '20231116'),
              'opto_food_week_bad': '202311_19_23',
              'opto_food_days_bad': ('20231119a', '20231119b', '20231121a', '20231121b', '20231123a', '20231123b'),
              'omission_week': '202312_03_05',
              'omission_days': ('20231203', '20231204', '20231205a', '20231205b'),
              'extinction_week': '202312_10_11',
              'extinction_days': ('20231210', '20231211'),
              'opto_test_days': ('20231213', '20231214'),
              'opto_test_week': '202312_13_14',
              'opto_food_week': '202312_24_28',
              'opto_food_days': ('20231224a', '20231224b', '20231226a', '20231226b', '20231228a', '20231228b')
              }

MOUSE_YP84 = {'path': join(BASE_PATH, 'YP84'),
              'n_cells': {'week1': 514, 'opto_water': 735, 'opto_food': 520, 'predictions': 670},
              'location': {'bregma': -700, 'height': -3200, 'rotate': 317},
              'week1': '202308_26_30',
              'week1_days': ('20230826', '20230827', '20230828', '20230829a', '20230829b', '20230830'),
              'opto_water_week': '202309_04_06',
              'opto_water_days': ('20230904', '20230905', '20230906'),
              'opto_food_week': '202309_10_14',
              'opto_food_days': ('20230910a', '20230910b', '20230912a', '20230912b', '20230914a', '20230914b'),
              'omission_week': '202309_26_28',
              'omission_days': ('20230926', '20230927', '20230928a', '20230928b'),
              'extinction_week': '202310_01_02',
              'extinction_days': ('20231001', '20231002'),
              'opto_test_days': ('20231107', '20231108'),
              'opto_test_week': '202311_07_08'
              }

MOUSE_YP86 = {'path': join(BASE_PATH, 'YP86'),
              'n_cells': {'week1': 333, 'opto_water': 425, 'opto_food': 306, 'predictions': 379},
              'location': {'bregma': -780, 'height': -3400, 'rotate': 323},
              'week1': '202308_26_30',
              'week1_days': ('20230826', '20230827', '20230828', '20230829a', '20230829b', '20230830'),
              'opto_water_week': '202309_04_06',
              'opto_water_days': ('20230904', '20230905', '20230906'),
              'opto_food_week': '202309_10_14',
              'opto_food_days': ('20230910a', '20230910b', '20230912a', '20230912b', '20230914a', '20230914b'),
              'omission_week': '202310_01_03',
              'omission_days': ('20231001', '20231002', '20231003a', '20231003b'),
              'extinction_week': '202311_01_02',
              'extinction_days': ('20231101', '20231102'),
              'opto_test_days': ('20231105', '20231106'),
              'opto_test_week': '202311_05_06',
              'opto_test_thirsty_days': ('20240121', '20240122'),
              'opto_test_thirsty_week': '202401_21_22'
              }


ALL_MICE = [MOUSE1, MOUSE2, MOUSE3, MOUSE_SS4, MOUSE_SS6, MOUSE_SS37, MOUSE_SS38, MOUSE_YP64, MOUSE_YP79,
            MOUSE_YP81, MOUSE_YP82, MOUSE_YP83, MOUSE_YP84, MOUSE_YP86]
OPTO_MICE = [MOUSE_YP64, MOUSE_YP79, MOUSE_YP81, MOUSE_YP82, MOUSE_YP83, MOUSE_YP84, MOUSE_YP86]
OPTO_CONTROL_MICE = [MOUSE_SS6, MOUSE_SS37, MOUSE_SS38]
ALL_OPTO_RELEVANT = [MOUSE_YP64, MOUSE_YP79, MOUSE_YP81, MOUSE_YP82, MOUSE_YP83, MOUSE_YP84, MOUSE_YP86, MOUSE_SS6,
                     MOUSE_SS37, MOUSE_SS38]
OMISSION_MICE = [MOUSE_SS4, MOUSE_YP64, MOUSE_YP79, MOUSE_YP82, MOUSE_YP83, MOUSE_YP84, MOUSE_YP86]
NEW_MICE = [MOUSE_SS4, MOUSE_SS6, MOUSE_SS37, MOUSE_SS38, MOUSE_YP64, MOUSE_YP79, MOUSE_YP81, MOUSE_YP82, MOUSE_YP83,
            MOUSE_YP84, MOUSE_YP86]
