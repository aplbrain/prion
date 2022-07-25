from AblationFunctions import *

token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" \
        ".eyJlbWFpbCI6ImxheDE4Y2hyaXN0aWFuQGdtYWlsLmNvbSIsImxldmVsIjoibm9hdXRoIiwiaW1hZ2UtdXJsIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUFUWEFKd002NG9MQmtBYnc3T1JES2JQbmRHWGRNdDgxMEE1ZUxUTGphUno9czk2LWM_c3o9NTA_c3o9NTAiLCJleHAiOjE4MzY1NzYzMDJ9.mvOOHTXT5emyB2yzh47uNmwYZL1SQOdoWgkFHuajVuU "

client = neuprint.Client('https://neuprint.janelia.org', token=token, dataset='hemibrain:v1.2.1')

w_thresh = 5
is_thresh = f"""AND w.weightHP >= {w_thresh}"""
types = ["PN", "KC", "APL", "MBON", "ORN"]  # neuron types in the MB
wheres_thresh = []

table = pd.DataFrame()
for c1 in types:
    for c2 in types:
        where = f"""(a.type CONTAINS "{c1}") AND (b.type CONTAINS "{c2}") {is_thresh}"""
        q = f" MATCH (a :`hemibrain_Neuron`)-[w:ConnectsTo]->(b:`hemibrain_Neuron`) WHERE {where} RETURN a.bodyId, " \
            f"a.type, b.bodyId, b.type, w.weight "

        lh_table = neuprint.fetch_custom(q)
        lh_table["Supertype_pre"] = c1
        lh_table["Supertype_post"] = c2
        table = pd.concat([table, lh_table])
table = table.rename(columns={"w.weight": "weight"})

mb_graph, type_dict_mb, new_matrix = preprocess_df(table)

mb_wm_random = perform_ablation(graph=mb_graph, func=random_detachment, n_passes=1, weight_key='weight', decay_vector=np.zeros(1), threshold=0)
mb_wm_binomial, dv_binomial = perform_ablation(graph=mb_graph, func=binomial_detachment, n_passes=1, weight_key='weight', decay_vector=np.zeros(1), threshold=0)
mb_wm_preferential, dv_preferential = perform_ablation(graph=mb_graph, func=preferential_detachment, n_passes=1, weight_key='weight', decay_vector=np.zeros(1), threshold=0)
mb_wm_calcium = perform_ablation(graph=mb_graph, func=calcium_detachment, n_passes=10, weight_key='weight', decay_vector=0, threshold=100)
mb_wm_random2 = perform_ablation2(new_matrix, random_detachment, 1, np.zeros(1), 0)
mb_wm_binomial2, dv_binomial2 = perform_ablation2(new_matrix, binomial_detachment, 1, 0, 0)
mb_wm_preferential2, dv_preferential2 = perform_ablation2(new_matrix, preferential_detachment, 1, 0, 0)
mb_wm_calcium2 = perform_ablation2(new_matrix, calcium_detachment, 1, 0, 0, 1000)

print("hello")



