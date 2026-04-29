import pandas as pd
import numpy as np
import random
import json

def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two GPS points"""
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_gps(path):
    df = pd.read_csv(path)
    df = df.sort_values("elapsed_time_ts")
    return df


def find_common_points(query, ref, dist_thresh=10):
    """Find GPS pairs that are spatially close"""
    pairs = []

    for i, q in query.iterrows():
        for j, r in ref.iterrows():
            d = haversine(q.latitude, q.longitude, r.latitude, r.longitude)
            if d < dist_thresh:
                pairs.append((i, j))

    return pairs


import random

def create_window(ts, length):
    """
    Create a window of given length where the timestamp ts
    appears at a random position inside the window.
    """

    # choose random offset where ts lies inside the window
    offset = random.uniform(0, length)

    start = ts - offset
    end = start + length

    return (start, end)




def generate_pairs(query_path, ref_path,
                   max_query_len,
                   max_ref_len,
                   n_pairs):

    query = load_gps(query_path)
    ref = load_gps(ref_path)

    matches = find_common_points(query, ref)

    results = []
    i = 0
    while len(results) < n_pairs and matches:
        
        qi, ri = random.choice(matches)

        q_ts = query.loc[qi].elapsed_time_ts
        q_gps = (query.loc[qi].latitude,query.loc[qi].longitude)
        r_ts = ref.loc[ri].elapsed_time_ts

        #q_len = random.uniform(1, max_query_len)
        #r_len = random.uniform(1, max_ref_len)
        q_len = max_query_len
        r_len = max_ref_len
        q_win = create_window(q_ts, q_len)
        r_win = create_window(r_ts, r_len)

        i = i+1
        results.append({"pair_id": i,
            "query_start": q_win[0],
            "query_end": q_win[1],
            "ref_start": r_win[0],
            "ref_end": r_win[1],
            "ref_l":r_win[1] - r_win[0],
            "query_l":q_win[1]-q_win[0],
            "ref_match": r_ts ,
            "q_match" : q_ts,
            "gps_q": q_gps,
            "match_type": "good"

        })

    return results


n_pairs=10
pairs = generate_pairs(
    query_path=r"C:\Arjun\Thesis\data\20200422_172431-sunset2\sunset2_gps_v2.csv",
    ref_path=r"C:\Arjun\Thesis\data\20200421_170039-sunset1\sunset1_gps_v2.csv",
    max_query_len=3, #in secs
    max_ref_len=7, #in secs
    n_pairs= 10
)
df = pd.DataFrame(pairs)




#config
output_config = r"C:\Arjun\Thesis\WF\src\dtw\mdtw_config.json"
config = {
            "experiment_name": f"dtw_test_{n_pairs}_pairs",
            "data": {
                "query_folder": "",
                "ref_folder": ""
            },
            "pairs": pairs
        }
with open(output_config, "w") as f:
    json.dump(config, f)

print("Run completed")
print(df[["pair_id", "query_l", "ref_l","gps_q"]].head(5))