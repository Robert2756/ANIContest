# Sequential Forward Selection
Mit Sequential Forward Selection (wrapper.py) wurden die folgenden Merkmale gefunden:
[225, 156, 132, 125, 58, 151, 172, 120, 117]

Hierbei wurde der beste BER Wert von 0.257 erziehlt (auf meinem Testdatensatz mit 350 Werten). Der Kassifikator wurde pro Durchgang 5 mal trainiert und der Durchschnitt gewählt.

# MIFS
Mithilfe des MIFS Algorithmus habe ich folgende Merkmale gefunden (MIFS.py).

[225, 8, 181, 180, 179, 178, 177, 176, 175]
Das Abbruchkriterium war diesmal ein MIFS Wert unter 0.

Hierbei wurde ein schlechtere BER Wer erziehlt.

# Adaboost
Es wurden bei jedem Training des Klassifikator immer die gleichen Samples falsch klassifiziert, was bedeutet dass die KLassifikatoren nicht divers genug waren um zu einem strong learner kombiniert zu werden.




