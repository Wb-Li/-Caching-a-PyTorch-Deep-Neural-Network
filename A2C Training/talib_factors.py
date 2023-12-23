import pandas as pd
import talib


def get_factors(df):
    tmp = df.copy()
    opening = df['open'].values
    closing = df['close'].values
    highest = df['high'].values
    lowest = df['low'].values
    volume = df['volume'].values
    
    # 累积/派发线（Accumulation / Distribution Line，该指标将每日的成交量通过价格加权累计，
    # 用以计算成交量的动量。属于趋势型因子
    tmp['AD'] = talib.AD(highest, lowest, closing, volume)

    # 佳庆指标（Chaikin Oscillator），该指标基于AD曲线的指数移动均线而计算得到。属于趋势型因子
    tmp['ADOSC'] = talib.ADOSC(highest, lowest, closing, volume, fastperiod=3, slowperiod=10)

    # 平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADX_5'] = talib.ADX(highest, lowest, closing, timeperiod=5)
    tmp['ADX_13'] = talib.ADX(highest, lowest, closing, timeperiod=13)
    tmp['ADX_21'] = talib.ADX(highest, lowest, closing, timeperiod=21)
    tmp['ADX_34'] = talib.ADX(highest, lowest, closing, timeperiod=34)
    tmp['ADX_55'] = talib.ADX(highest, lowest, closing, timeperiod=55)
    tmp['ADX_88'] = talib.ADX(highest, lowest, closing, timeperiod=88)
    tmp['ADX_120'] = talib.ADX(highest, lowest, closing, timeperiod=120)
    
    # 相对平均动向指数，DMI因子的构成部分。属于趋势型因子
    tmp['ADXR_5'] = talib.ADXR(highest, lowest, closing, timeperiod=5)
    tmp['ADXR_13'] = talib.ADXR(highest, lowest, closing, timeperiod=13)
    tmp['ADXR_21'] = talib.ADXR(highest, lowest, closing, timeperiod=21)
    tmp['ADXR_34'] = talib.ADXR(highest, lowest, closing, timeperiod=34)
    tmp['ADXR_55'] = talib.ADXR(highest, lowest, closing, timeperiod=55)
    tmp['ADXR_88'] = talib.ADXR(highest, lowest, closing, timeperiod=88)
    tmp['ADXR_120'] = talib.ADXR(highest, lowest, closing, timeperiod=120)
    
    # 绝对价格振荡指数
    tmp['APO'] = talib.APO(closing, fastperiod=12, slowperiod=26)

    # Aroon通过计算自价格达到近期最高值和最低值以来所经过的期间数，
    # 帮助投资者预测证券价格从趋势到区域区域或反转的变化，
    # Aroon指标分为Aroon、AroonUp和AroonDown3个具体指标。属于趋势型因子
    tmp['AROONDown_5'], tmp['AROONUp_5'] = talib.AROON(highest, lowest, timeperiod=5)
    tmp['AROONDown_13'], tmp['AROONUp_13'] = talib.AROON(highest, lowest, timeperiod=13)
    tmp['AROONDown_21'], tmp['AROONUp_21'] = talib.AROON(highest, lowest, timeperiod=21)
    tmp['AROONDown_34'], tmp['AROONUp_34'] = talib.AROON(highest, lowest, timeperiod=34)
    tmp['AROONDown_55'], tmp['AROONUp_55'] = talib.AROON(highest, lowest, timeperiod=55)
    tmp['AROONDown_88'], tmp['AROONUp_88'] = talib.AROON(highest, lowest, timeperiod=88)
    tmp['AROONDown_120'], tmp['AROONUp_120'] = talib.AROON(highest, lowest, timeperiod=120)
    
    tmp['AROONOSC_5'] = talib.AROONOSC(highest, lowest, timeperiod=5)
    tmp['AROONOSC_13'] = talib.AROONOSC(highest, lowest, timeperiod=13)
    tmp['AROONOSC_21'] = talib.AROONOSC(highest, lowest, timeperiod=21)
    tmp['AROONOSC_34'] = talib.AROONOSC(highest, lowest, timeperiod=34)
    tmp['AROONOSC_55'] = talib.AROONOSC(highest, lowest, timeperiod=55)
    tmp['AROONOSC_88'] = talib.AROONOSC(highest, lowest, timeperiod=88)
    tmp['AROONOSC_120'] = talib.AROONOSC(highest, lowest, timeperiod=120)

    # 均幅指标（Average TRUE Ranger），取一定时间周期内的股价波动幅度的移动平均值，
    # 是显示市场变化率的指标，主要用于研判买卖时机。属于超买超卖型因子。
    tmp['ATR_5'] = talib.ATR(highest, lowest, closing, timeperiod=5)
    tmp['ATR_13'] = talib.ATR(highest, lowest, closing, timeperiod=13)
    tmp['ATR_21'] = talib.ATR(highest, lowest, closing, timeperiod=21)
    tmp['ATR_34'] = talib.ATR(highest, lowest, closing, timeperiod=34)
    tmp['ATR_55'] = talib.ATR(highest, lowest, closing, timeperiod=55)
    tmp['ATR_88'] = talib.ATR(highest, lowest, closing, timeperiod=88)
    tmp['ATR_120'] = talib.ATR(highest, lowest, closing, timeperiod=120)
    
    # 布林带
    tmp['Boll_Up'], tmp['Boll_Mid'], tmp['Boll_Down'] = talib.BBANDS(closing,
        timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    # 均势指标
    tmp['BOP'] = talib.BOP(opening, highest, lowest, closing)

    # 5日顺势指标（Commodity Channel Index），专门测量股价是否已超出常态分布范围。属于超买超卖型因子。
    tmp['CCI_5'] = talib.CCI(highest, lowest, closing, timeperiod=5)
    tmp['CCI_13'] = talib.CCI(highest, lowest, closing, timeperiod=13)
    tmp['CCI_21'] = talib.CCI(highest, lowest, closing, timeperiod=21)
    tmp['CCI_34'] = talib.CCI(highest, lowest, closing, timeperiod=34)
    tmp['CCI_55'] = talib.CCI(highest, lowest, closing, timeperiod=55)
    tmp['CCI_88'] = talib.CCI(highest, lowest, closing, timeperiod=88)
    tmp['CCI_120'] = talib.CCI(highest, lowest, closing, timeperiod=120)
    
    # 钱德动量摆动指标（Chande Momentum Osciliator），与其他动量指标摆动指标如
    # 相对强弱指标（RSI）和随机指标（KDJ）不同，
    # 钱德动量指标在计算公式的分子中采用上涨日和下跌日的数据。属于超买超卖型因子
    tmp['CMO_Close_5'] = talib.CMO(closing, timeperiod=5)
    tmp['CMO_Close_13'] = talib.CMO(closing, timeperiod=13)
    tmp['CMO_Close_21'] = talib.CMO(closing, timeperiod=21)
    tmp['CMO_Close_34'] = talib.CMO(closing, timeperiod=34)
    tmp['CMO_Close_55'] = talib.CMO(closing, timeperiod=55)
    tmp['CMO_Close_88'] = talib.CMO(closing, timeperiod=88)
    tmp['CMO_Close_120'] = talib.CMO(closing, timeperiod=120)
    
    tmp['CMO_volume_5'] = talib.CMO(volume, timeperiod=5)
    tmp['CMO_volume_13'] = talib.CMO(volume, timeperiod=13)
    tmp['CMO_volume_21'] = talib.CMO(volume, timeperiod=21)
    tmp['CMO_volume_34'] = talib.CMO(volume, timeperiod=34)
    tmp['CMO_volume_55'] = talib.CMO(volume, timeperiod=55)
    tmp['CMO_volume_88'] = talib.CMO(volume, timeperiod=88)
    tmp['CMO_volume_120'] = talib.CMO(volume, timeperiod=120)
    
    #MACD
    tmp['macd'], tmp['macdsignal'], tmp['macdhist'] = talib.MACD(closing, 
                                                                 fastperiod=12, 
                                                                 slowperiod=26, 
                                                                 signalperiod=9)
    
    # DEMA双指数移动平均线
    tmp['DEMA_5'] = talib.DEMA(closing, timeperiod=5)
    tmp['DEMA_13'] = talib.DEMA(closing, timeperiod=13)
    tmp['DEMA_21'] = talib.DEMA(closing, timeperiod=21)
    tmp['DEMA_34'] = talib.DEMA(closing, timeperiod=34)
    tmp['DEMA_55'] = talib.DEMA(closing, timeperiod=55)
    tmp['DEMA_88'] = talib.DEMA(closing, timeperiod=88)
    tmp['DEMA_120'] = talib.DEMA(closing, timeperiod=120)

    # DX 动向指数
    tmp['DX_5'] = talib.DX(highest, lowest, closing, timeperiod=5)
    tmp['DX_13'] = talib.DX(highest, lowest, closing, timeperiod=13)
    tmp['DX_21'] = talib.DX(highest, lowest, closing, timeperiod=21)
    tmp['DX_34'] = talib.DX(highest, lowest, closing, timeperiod=34)
    tmp['DX_55'] = talib.DX(highest, lowest, closing, timeperiod=55)
    tmp['DX_88'] = talib.DX(highest, lowest, closing, timeperiod=88)
    tmp['DX_120'] = talib.DX(highest, lowest, closing, timeperiod=120)
    
    # EMA 指数移动平均线
    tmp['EMA_5'] = talib.EMA(closing, timeperiod=5)
    tmp['EMA_13'] = talib.EMA(closing, timeperiod=13)
    tmp['EMA_21'] = talib.EMA(closing, timeperiod=21)
    tmp['EMA_34'] = talib.EMA(closing, timeperiod=34)
    tmp['EMA_55'] = talib.EMA(closing, timeperiod=55)
    tmp['EMA_88'] = talib.EMA(closing, timeperiod=88)
    tmp['EMA_120'] = talib.EMA(closing, timeperiod=120)

    # KAMA 适应性移动平均线
    tmp['KAMA_5'] = talib.KAMA(closing, timeperiod=5)
    tmp['KAMA_13'] = talib.KAMA(closing, timeperiod=13)
    tmp['KAMA_21'] = talib.KAMA(closing, timeperiod=21)
    tmp['KAMA_34'] = talib.KAMA(closing, timeperiod=34)
    tmp['KAMA_55'] = talib.KAMA(closing, timeperiod=55)
    tmp['KAMA_88'] = talib.KAMA(closing, timeperiod=88)
    tmp['KAMA_120'] = talib.KAMA(closing, timeperiod=120)
        
    #WMA
    tmp['WMA_5'] = talib.WMA(closing, timeperiod=5)
    tmp['WMA_13'] = talib.WMA(closing, timeperiod=13)
    tmp['WMA_21'] = talib.WMA(closing, timeperiod=21)
    tmp['WMA_34'] = talib.WMA(closing, timeperiod=34)
    tmp['WMA_55'] = talib.WMA(closing, timeperiod=55)
    tmp['WMA_88'] = talib.WMA(closing, timeperiod=88)
    tmp['WMA_120'] = talib.WMA(closing, timeperiod=120)
    
    #HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
    tmp['HT_DCPERIOD'] = talib.HT_DCPERIOD(closing)
    
    #HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
    tmp['HT_DCPHASE'] = talib.HT_DCPHASE(closing)
    
    #HT_PHASOR - Hilbert Transform - Phasor Components
    tmp['inphase'],  tmp['quadrature'] = talib.HT_PHASOR(closing)
    
    #HT_SINE - Hilbert Transform - SineWave
    tmp['sine'],  tmp['leadsine'] = talib.HT_SINE(closing)
    
    #HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
    tmp['integer'] = talib.HT_TRENDMODE(closing)
    
    #MAMA
    tmp['mama'], tmp['fama'] = talib.MAMA(closing, fastlimit=0, slowlimit=0)
        
    #STOCH - Stochastic
    tmp['slowk'], tmp['slowd'] = talib.STOCH(highest, lowest, closing, fastk_period=5, slowk_period=3, 
                                             slowk_matype=0, slowd_period=3, slowd_matype=0)

    #STOCHF - Stochastic Fast
    tmp['fastk'], tmp['fastd'] = talib.STOCHF(highest, lowest, closing, fastk_period=5, fastd_period=3, fastd_matype=0)
    
    #STOCHRSI - Stochastic Relative Strength Index
    tmp['fastk'], tmp['fastd'] = talib.STOCHRSI(closing, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    
    # MACD
    tmp['MACD_DIF'], tmp['MACD_DEA'], tmp['MACD_bar'] = talib.MACD(closing, fastperiod=12, slowperiod=24, signalperiod=9)
    
    #ULTOSC - Ultimate Oscillator
    tmp['ULTOSC'] = talib.ULTOSC(highest, lowest, closing, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    
    #MFI
    tmp['MFI_5'] = talib.MFI(highest, lowest, closing, volume, timeperiod=5)
    tmp['MFI_13'] = talib.MFI(highest, lowest, closing, volume, timeperiod=13)
    tmp['MFI_21'] = talib.MFI(highest, lowest, closing, volume, timeperiod=21)
    tmp['MFI_34'] = talib.MFI(highest, lowest, closing, volume, timeperiod=34)
    tmp['MFI_55'] = talib.MFI(highest, lowest, closing, volume, timeperiod=55)
    tmp['MFI_88'] = talib.MFI(highest, lowest, closing, volume, timeperiod=88)
    tmp['MFI_120'] = talib.MFI(highest, lowest, closing, volume, timeperiod=120)
    
    # 中位数价格
    tmp['MEDPRICE'] = talib.MEDPRICE(highest, lowest)

    # 负向指标 负向运动
    tmp['MiNUS_DI_5'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=5)
    tmp['MiNUS_DI_13'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=13)
    tmp['MiNUS_DI_21'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=21)
    tmp['MiNUS_DI_34'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=34)
    tmp['MiNUS_DI_55'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=55)
    tmp['MiNUS_DI_88'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=88)
    tmp['MiNUS_DI_120'] = talib.MINUS_DI(highest, lowest, closing, timeperiod=120)
    
    tmp['MiNUS_DM_5'] = talib.MINUS_DM(highest, lowest, timeperiod=5)
    tmp['MiNUS_DM_13'] = talib.MINUS_DM(highest, lowest, timeperiod=13)
    tmp['MiNUS_DM_21'] = talib.MINUS_DM(highest, lowest, timeperiod=24)
    tmp['MiNUS_DM_34'] = talib.MINUS_DM(highest, lowest, timeperiod=34)
    tmp['MiNUS_DM_55'] = talib.MINUS_DM(highest, lowest, timeperiod=55)
    tmp['MiNUS_DM_88'] = talib.MINUS_DM(highest, lowest, timeperiod=88)
    tmp['MiNUS_DM_120'] = talib.MINUS_DM(highest, lowest, timeperiod=120)

    # 动量指标（Momentom Index），动量指数以分析股价波动的速度为目的，研究股价在波动过程中各种加速，
    # 减速，惯性作用以及股价由静到动或由动转静的现象。属于趋势型因子
    tmp['MOM_5'] = talib.MOM(closing, timeperiod=5)
    tmp['MOM_13'] = talib.MOM(closing, timeperiod=13)
    tmp['MOM_21'] = talib.MOM(closing, timeperiod=21)
    tmp['MOM_34'] = talib.MOM(closing, timeperiod=34)
    tmp['MOM_55'] = talib.MOM(closing, timeperiod=55)
    tmp['MOM_88'] = talib.MOM(closing, timeperiod=88)
    tmp['MOM_120'] = talib.MOM(closing, timeperiod=120)

    # 归一化平均值范围
    tmp['NATR_5'] = talib.NATR(highest, lowest, closing, timeperiod=5)
    tmp['NATR_13'] = talib.NATR(highest, lowest, closing, timeperiod=13)
    tmp['NATR_21'] = talib.NATR(highest, lowest, closing, timeperiod=21)
    tmp['NATR_34'] = talib.NATR(highest, lowest, closing, timeperiod=34)
    tmp['NATR_55'] = talib.NATR(highest, lowest, closing, timeperiod=55)
    tmp['NATR_88'] = talib.NATR(highest, lowest, closing, timeperiod=88)
    tmp['NATR_120'] = talib.NATR(highest, lowest, closing, timeperiod=120)

    # OBV 	能量潮指标（On Balance Volume，OBV），以股市的成交量变化来衡量股市的推动力，
    # 从而研判股价的走势。属于成交量型因子
    tmp['OBV'] = talib.OBV(closing, volume)

    # PLUS_DI 更向指示器
    tmp['PLUS_DI_5'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=5)
    tmp['PLUS_DI_13'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=13)
    tmp['PLUS_DI_21'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=21)
    tmp['PLUS_DI_34'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=34)
    tmp['PLUS_DI_55'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=55)
    tmp['PLUS_DI_88'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=88)
    tmp['PLUS_DI_120'] = talib.PLUS_DI(highest, lowest, closing, timeperiod=120)
    
    tmp['PLUS_DM_5'] = talib.PLUS_DM(highest, lowest, timeperiod=5)
    tmp['PLUS_DM_13'] = talib.PLUS_DM(highest, lowest, timeperiod=13)
    tmp['PLUS_DM_21'] = talib.PLUS_DM(highest, lowest, timeperiod=21)
    tmp['PLUS_DM_34'] = talib.PLUS_DM(highest, lowest, timeperiod=34)
    tmp['PLUS_DM_55'] = talib.PLUS_DM(highest, lowest, timeperiod=55)
    tmp['PLUS_DM_88'] = talib.PLUS_DM(highest, lowest, timeperiod=88)
    tmp['PLUS_DM_120'] = talib.PLUS_DM(highest, lowest, timeperiod=120)

    # PPO 价格振荡百分比
    tmp['PPO'] = talib.PPO(closing, fastperiod=6, slowperiod=26, matype=0)

    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    #tmp['ROC6'] = talib.ROC(closing, timeperiod=6)
    #tmp['ROC20'] = talib.ROC(closing, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    tmp['VROC_5'] = talib.ROC(volume, timeperiod=5)
    tmp['VROC_13'] = talib.ROC(volume, timeperiod=13)
    tmp['VROC_21'] = talib.ROC(volume, timeperiod=21)
    tmp['VROC_34'] = talib.ROC(volume, timeperiod=34)
    tmp['VROC_55'] = talib.ROC(volume, timeperiod=55)
    tmp['VROC_88'] = talib.ROC(volume, timeperiod=88)
    tmp['VROC_120'] = talib.ROC(volume, timeperiod=120)
    
    # ROC 6日变动速率（Price Rate of Change），以当日的收盘价和N天前的收盘价比较，
    # 通过计算股价某一段时间内收盘价变动的比例，应用价格的移动比较来测量价位动量。属于超买超卖型因子。
    #tmp['ROCP6'] = talib.ROCP(closing, timeperiod=6)
    #tmp['ROCP20'] = talib.ROCP(closing, timeperiod=20)
    # 12日量变动速率指标（Volume Rate of Change），以今天的成交量和N天前的成交量比较，
    # 通过计算某一段时间内成交量变动的幅度，应用成交量的移动比较来测量成交量运动趋向，
    # 达到事先探测成交量供需的强弱，进而分析成交量的发展趋势及其将来是否有转势的意愿，
    # 属于成交量的反趋向指标。属于成交量型因子
    tmp['VROCP5'] = talib.ROCP(volume, timeperiod=5)
    tmp['VROCP13'] = talib.ROCP(volume, timeperiod=13)
    tmp['VROCP21'] = talib.ROCP(volume, timeperiod=21)
    tmp['VROCP34'] = talib.ROCP(volume, timeperiod=34)
    tmp['VROCP55'] = talib.ROCP(volume, timeperiod=55)
    tmp['VROCP88'] = talib.ROCP(volume, timeperiod=88)
    tmp['VROCP120'] = talib.ROCP(volume, timeperiod=120)

    # RSI
    tmp['RSI_5'] = talib.RSI(closing, timeperiod=5)
    tmp['RSI_13'] = talib.RSI(closing, timeperiod=13)
    tmp['RSI_21'] = talib.RSI(closing, timeperiod=21)
    tmp['RSI_34'] = talib.RSI(closing, timeperiod=34)
    tmp['RSI_55'] = talib.RSI(closing, timeperiod=55)
    tmp['RSI_88'] = talib.RSI(closing, timeperiod=88)
    tmp['RSI_120'] = talib.RSI(closing, timeperiod=120)
    
    # SAR 抛物线转向
    tmp['SAR'] = talib.SAR(highest, lowest, acceleration=0.02, maximum=0.2)

    # TEMA
    tmp['TEMA5'] = talib.TEMA(closing, timeperiod=5)
    tmp['TEMA13'] = talib.TEMA(closing, timeperiod=13)
    tmp['TEMA21'] = talib.TEMA(closing, timeperiod=21)
    tmp['TEMA34'] = talib.TEMA(closing, timeperiod=34)
    tmp['TEMA55'] = talib.TEMA(closing, timeperiod=55)
    tmp['TEMA88'] = talib.TEMA(closing, timeperiod=88)
    tmp['TEMA120'] = talib.TEMA(closing, timeperiod=120)

    # TRANGE 真实范围
    tmp['TRANGE'] = talib.TRANGE(highest, lowest, closing)

    # TYPPRICE 典型价格
    tmp['TYPPRICE'] = talib.TYPPRICE(highest, lowest, closing)

    # TSF 时间序列预测
    tmp['TSF_5'] = talib.TSF(closing, timeperiod=5)
    tmp['TSF_13'] = talib.TSF(closing, timeperiod=13)
    tmp['TSF_21'] = talib.TSF(closing, timeperiod=24)
    tmp['TSF_34'] = talib.TSF(closing, timeperiod=34)
    tmp['TSF_55'] = talib.TSF(closing, timeperiod=55)
    tmp['TSF_88'] = talib.TSF(closing, timeperiod=88)
    tmp['TSF_120'] = talib.TSF(closing, timeperiod=120)
    
    # ULTOSC 极限振子
    tmp['ULTOSC'] = talib.ULTOSC(highest, lowest, closing, timeperiod1=7, 
                                 timeperiod2=14, timeperiod3=28)

    # 威廉指标
    tmp['WILLR_5'] = talib.WILLR(highest, lowest, closing, timeperiod=5)
    tmp['WILLR_13'] = talib.WILLR(highest, lowest, closing, timeperiod=13)
    tmp['WILLR_21'] = talib.WILLR(highest, lowest, closing, timeperiod=21)
    tmp['WILLR_34'] = talib.WILLR(highest, lowest, closing, timeperiod=34)
    tmp['WILLR_55'] = talib.WILLR(highest, lowest, closing, timeperiod=55)
    tmp['WILLR_88'] = talib.WILLR(highest, lowest, closing, timeperiod=88)
    tmp['WILLR_120'] = talib.WILLR(highest, lowest, closing, timeperiod=120)
    
    return tmp