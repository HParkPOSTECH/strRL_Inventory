adap_rate = {"1SRL-PSAstationarygamma": {'tau': 1.0, 'sigma': 1.0, 'a_rate': 0.01, 'eta_denom': 0.01, 'eta_nom': 1,'eta_mul': 0.7, 'b1_denom': 0.01, 'b1_nom': 10, 'b1_mul': 1., 'b2_denom': 0.1, 'b2_nom': 20, 'b2_mul': 0.9, 'mt': 0.95},
             "1SRL-PSAstationarynormal": {'tau': 1.0, 'sigma': 1.0, 'a_rate': 0.01, 'eta_denom': 0.01, 'eta_nom': 10,'eta_mul': 0.6, 'b1_denom': 0.01, 'b1_nom': 10, 'b1_mul': 1., 'b2_denom': 0.1, 'b2_nom': 10, 'b2_mul': 0.7, 'mt': 0.95},
            "1SRL-PSAstationarypoisson": {'tau': 1.5, 'sigma': 2.0, 'a_rate': 0.05, 'eta_denom': 0.25, 'eta_nom': 20,'eta_mul': 0.6, 'b1_denom': 0.5, 'b1_nom': 10, 'b1_mul': 0.9, 'b2_denom': 0.1, 'b2_nom': 10, 'b2_mul': 0.8, 'mt': 0.9},
            "1SRL-PSAnon-stationarygamma": {'tau': 1.0, 'sigma': 2.0, 'a_rate': 0.5, 'eta_denom': 0.002, 'eta_nom': 5,'eta_mul': 0.6, 'b1_denom': 0.01, 'b1_nom': 20, 'b1_mul': 0.9, 'b2_denom': 0.01, 'b2_nom': 20, 'b2_mul': 0.8, 'mt': 0.05},
             "1SRL-FSAstationarygamma": {'tau': 1.5, 'sigma': 2.0, 'a_rate': 0.01, 'eta_denom': 0.1, 'eta_nom': 1,'eta_mul': 0.8, 'b1_denom': 0.1, 'b1_nom': 10, 'b1_mul': 1., 'b2_denom': 0.01, 'b2_nom': 10, 'b2_mul': 0.8, 'm': 50, 'mt': 0.95},
             "1SRL-FSAstationarynormal": {'tau': 1.0, 'sigma': 1.0, 'a_rate': 0.01, 'eta_denom': 0.01, 'eta_nom': 10,'eta_mul': 0.6, 'b1_denom': 0.01, 'b1_nom': 10, 'b1_mul': 0.7, 'b2_denom': 0.1, 'b2_nom': 10, 'b2_mul': 0.7, 'm': 50, 'mt': 0.95},
             "1SRL-FSAstationarypoisson": {'tau': 2.5, 'sigma': 0.5, 'a_rate': 0.5, 'eta_denom': 0.05, 'eta_nom': 10,'eta_mul': 0.6, 'b1_denom': 1.0, 'b1_nom': 20, 'b1_mul': 0.7, 'b2_denom': 1.0, 'b2_nom': 10, 'b2_mul': 0.7, 'm': 10, 'mt': 0.3},
             "1SRL-FSAnon-stationarygamma": {'tau': 0.5, 'sigma': 0.5, 'a_rate': 0.5, 'eta_denom': 0.01, 'eta_nom': 20,'eta_mul': 0.6, 'b1_denom': 0.0015, 'b1_nom': 20, 'b1_mul': 0.8, 'b2_denom': 0.002, 'b2_nom': 20, 'b2_mul': 1.0, 'm': 40, 'mt': 0.05},
             "2SRL-PSAstationarygamma": {'tau': 1.0, 'sigma': 1.5, 'a_rate': 0.01, 'eta_denom': 0.01, 'eta_nom': 10, 'eta_mul': 0.6, 'b1_denom': 0.004, 'b1_nom': 10, 'b1_mul': 0.85, 'b2_denom': 0.03, 'b2_nom': 20, 'b2_mul': 0.7, 'b3_denom': 0.02, 'b3_nom': 20, 'b3_mul': 0.8, 'mt': 0.9},
             "2SRL-PSAnon-stationarygamma": {'tau': 1.0, 'sigma': 1.0, 'a_rate': 0.1, 'eta_denom': 0.00001, 'eta_nom': 10, 'eta_mul': 0.6, 'b1_denom': 0.005, 'b1_nom': 10, 'b1_mul': 1.0, 'b2_denom': 0.005, 'b2_nom': 20, 'b2_mul': 0.8, 'b3_denom': 0.008, 'b3_nom': 20, 'b3_mul': 1.0, 'mt': 0.5},
             "2SRL-FSAstationarygamma": {'tau': 1.5, 'sigma': 2.0,  'a_rate': 0.05, 'eta_denom': 0.1, 'eta_nom': 10, 'eta_mul': 0.6, 'b1_denom': 0.5, 'b1_nom': 5, 'b1_mul': 1.0,'b2_denom': 0.1, 'b2_nom': 5, 'b2_mul': 0.8, 'b3_denom': 0.1, 'b3_nom': 20, 'b3_mul': 0.8, 'mS': 128, 'mc': 150, 'ms': 30, 'mt': 0.5},
             "2SRL-FSAnon-stationarygamma": {'tau': 0.3, 'sigma': 0.3,  'a_rate': 0.1, 'eta_denom': 0.000001, 'eta_nom': 10, 'eta_mul': 0.6, 'b1_denom': 0.02, 'b1_nom': 10, 'b1_mul': 0.8, 'b2_denom': 0.02, 'b2_nom': 10, 'b2_mul': 0.9, 'b3_denom': 0.02, 'b3_nom': 10, 'b3_mul': 0.9, 'mS': 200, 'mc': 200, 'ms': 100, 'mt': 0.5},
             "3SRL-PSAstationarygamma": {'tau': 2.0, 'sigma': 2.0, 'a_rate': 0.001, 'eta_denom': 0.0005, 'eta_nom': 10, 'eta_mul': 0.6, 'b1_denom': 0.1, 'b1_nom': 10, 'b1_mul': 1.0, 'b2_denom': 0.1, 'b2_nom': 10, 'b2_mul': 0.7, 'b3_denom': 0.1, 'b3_nom': 10, 'b3_mul': 0.7, 'mt': 0.9},
             "3SRL-FSAstationarygamma": {'tau': 1.0, 'sigma': 1.0, 'a_rate': 0.001, 'eta_denom': 0.0008, 'eta_nom': 20, 'eta_mul': 0.6, 'b1_denom': 2.0, 'b1_nom': 10, 'b1_mul': 0.8, 'b2_denom': 5.0, 'b2_nom': 10, 'b2_mul': 0.7, 'b3_denom': 4.0, 'b3_nom': 10, 'b3_mul': 0.8, 'mS': 1, 'mc': 150, 'ms': 150, 'mt': 0.9},
             "4SRL-PSAstationarygamma": {'tau': 2.0, 'sigma': 2.0, 'a_rate': 0.001, 'eta_denom': 0.00075, 'eta_nom': 10, 'eta_mul': 0.6, 'b1_denom': 0.1, 'b1_nom': 10, 'b1_mul': 1.0, 'b2_denom': 0.1, 'b2_nom': 10, 'b2_mul': 0.7, 'b3_denom': 0.3, 'b3_nom': 10, 'b3_mul': 0.7, 'mt': 0.9},
             "4SRL-FSAstationarygamma": {'tau': 2.0, 'sigma': 1.5, 'a_rate': 0.0075, 'eta_denom': 0.0007, 'eta_nom': 10, 'eta_mul': 0.6, 'b1_denom': 2.5, 'b1_nom': 10, 'b1_mul': 0.7, 'b2_denom': 4.0, 'b2_nom': 20, 'b2_mul': 0.7, 'b3_denom': 4.0, 'b3_nom': 10, 'b3_mul': 0.7, 'mS': 1, 'mc': 250, 'ms': 250, 'mt': 0.7}
             }  # adaptation rate setting for each numerical study

init_t = 0 # initial period
threshold_multiplier = 2.3  # threshold for ratio change
max_len = 100  # maximum observation queue length
h = 5. # normalized factor value
value_degree = 4  # degree of value function
bigM = 1000.  # large constant
low_pos = 1e-6  # low positive value