import pandas as pd
import numpy as np


def get_venus() -> dict:
    filepaths = [
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\Venüs\Venüs Finansallar ve KPI_lar.xlsx",
    ]
    data = pd.read_excel(filepaths[0], skiprows=[1])
    data_header : list = data.head(0).columns.values.tolist()
    data_as_list : list = data.values.tolist()
    data_as_list.insert(0, data_header)

    data_as_list = [[h if not pd.isna(h) else "" for h in i][:12] for i in data_as_list]
    data_as_list = [i for i in data_as_list if i != [''] * len(i)]
    data_as_dict = {k: [*v] for [k, *v] in data_as_list}
    data_as_dict = {k: v for k, v in data_as_dict.items() if v != [''] * len(v)}
    return data_as_dict


def get_saturn() -> dict:
    filepaths = [
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\Satürn\Satürn KPI ve Finansallar.xlsx",
    ]
    data = pd.read_excel(filepaths[0])
    data_header : list = data.head(0).columns.values.tolist()
    data_as_list : list = data.values.tolist()
    data_as_list.insert(0, data_header)

    data_as_list = [[h if not pd.isna(h) else "" for h in i][:10] for i in data_as_list]
    data_as_list = [i for i in data_as_list if i != [''] * len(i)]
    data_as_dict = {k: [*v] for [k, *v] in data_as_list}
    data_as_dict = {k: v for k, v in data_as_dict.items() if v != [''] * len(v)}
    return data_as_dict


def get_neptun() -> dict:
    filepaths = [
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\Neptün\Neptün KPI ve Finansallar.xlsx",
    ]
    data = pd.read_excel(filepaths[0])
    data_header : list = data.head(0).columns.values.tolist()
    data_as_list : list = data.values.tolist()
    data_as_list.insert(0, data_header)

    data_as_list = [[h if not pd.isna(h) else "" for h in i][:10] for i in data_as_list]
    data_as_list = [i for i in data_as_list if i != [''] * len(i)]
    data_as_dict = {k: [*v] for [k, *v] in data_as_list}
    data_as_dict = {k: v for k, v in data_as_dict.items() if v != [''] * len(v)}
    return data_as_dict


def get_mars() -> dict:
    filepaths = [
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\MARS\Mars Finansallar ve KPI lar.xlsx",
    ]
    data = pd.read_excel(filepaths[0])
    data_header : list = data.head(0).columns.values.tolist()
    data_as_list : list = data.values.tolist()
    data_as_list.insert(0, data_header)

    data_as_list = [[h if not pd.isna(h) else "" for h in i][:10] for i in data_as_list]
    data_as_list = [i for i in data_as_list if i != [''] * len(i)]
    data_as_dict = {k: [*v] for [k, *v] in data_as_list}
    data_as_dict = {k: v for k, v in data_as_dict.items() if v != [''] * len(v)}
    return data_as_dict


def get_jupiter() -> dict:
    filepaths = [
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\Jüpiter\Jupiter Financiasl ve KPI_lar.xlsx",
    ]
    data = pd.read_excel(filepaths[0])
    data_header : list = data.head(0).columns.values.tolist()
    data_as_list : list = data.values.tolist()
    data_as_list.insert(0, data_header)

    data_as_list = [[h if not pd.isna(h) else "" for h in i][:10] for i in data_as_list]
    data_as_list = [i for i in data_as_list if i != [''] * len(i)]
    data_as_dict = {k: [*v] for [k, *v] in data_as_list}
    data_as_dict = {k: v for k, v in data_as_dict.items() if v != [''] * len(v)}
    return data_as_dict


def get_dunya() -> dict:
    filepaths = [
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\Dünya\Dünya Finansallar.xlsx",
    ]
    data = pd.read_excel(filepaths[0])
    data_header : list = data.head(0).columns.values.tolist()
    data_as_list : list = data.values.tolist()
    data_as_list.insert(0, data_header)

    data_as_list = [[h if not pd.isna(h) else "" for h in i][:10] for i in data_as_list]
    data_as_list = [i for i in data_as_list if i != [''] * len(i)]
    data_as_dict = {k: [*v] for [k, *v] in data_as_list}
    data_as_dict = {k: v for k, v in data_as_dict.items() if v != [''] * len(v)}
    return data_as_dict


def get_hushhush() -> dict:
    filepaths = [
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\HushHush Games\HushHush - Business Plan.xlsx",
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\HushHush Games\KPI Table.xlsx",
    ]
    data = pd.read_excel(filepaths[0], skiprows=[1])
    data_header : list = data.head(0).columns.values.tolist()
    data_as_list : list = data.values.tolist()
    data_as_list.insert(0, data_header)

    data_as_list = [[h if not pd.isna(h) else "" for h in i][:13] for i in data_as_list]
    data_as_list = [i for i in data_as_list if i != [''] * len(i)]
    return data_as_list


def get_soulbound() -> dict:
    filepaths = [
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\SOULBOUND Games\Project Sprint KPI - Final.xlsx",
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\SOULBOUND Games\Project Sprint_2022_Monthly P_L__Nov22 Final.xlsx",
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\SOULBOUND Games\SOULBOUND Games - Metrik Karşılaştırması.xlsx",
    ]
    data = pd.read_excel(filepaths[0], skiprows=[1])
    data_header : list = data.head(0).columns.values.tolist()
    data_as_list : list = data.values.tolist()
    data_as_list.insert(0, data_header)

    data_as_list = [[h if not pd.isna(h) else "" for h in i][:13] for i in data_as_list]
    data_as_list = [i for i in data_as_list if i != [''] * len(i)]
    return data_as_list


def get_agamemnon() -> dict:
    filepaths = [
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\Agamemnon Games\Agamemnon Games Annual Projections - December 2022.xlsx",
        r"C:\Users\umut_\OneDrive\Desktop\drive-download-20230327T131659Z-001\Agamemnon Games\Agamemnon Games FB Data Summary - 18.12.2022.xlsx",
    ]
    data = pd.read_excel(filepaths[0])
    data_header : list = data.head(0).columns.values.tolist()
    data_as_list : list = data.values.tolist()
    data_as_list.insert(0, data_header)

    data_as_list = [[h if not pd.isna(h) else "" for h in i][:13] for i in data_as_list]
    data_as_list = [i for i in data_as_list if i != [''] * len(i)]
    data_as_dict = {k: [*v] for [k, *v] in data_as_list}
    data_as_dict = {k: v for k, v in data_as_dict.items() if v != [''] * len(v)}
    return data_as_dict


print("\nAGAMEMNON")
for k, v in get_agamemnon().items():
    print(f"{k} ---> {[round(h, 2) if isinstance(h, float) or isinstance(h, int) else h for h in v]}")
    