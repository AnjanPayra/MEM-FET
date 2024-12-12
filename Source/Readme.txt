
Read me details:

1.bagging_Regressor.py - predict using BaggingRegressor model
input-protein wise CC, ECC, functinal activity score and MEM-FET score.
output-predicting the output accuracy and mean_squared_error value.

2. Classifier.py -Diff_Classifier to find essential properties of the dataset
input-All centrality measures , CC, ECC, Functional Activity score, Localized significance score and physico-chemical measures.
output-Find selected properties using classifiers.

3. Ensemble.py
input-
output-

3. esemble_avg_reg.py
input-
output-

5. Esemble_bag_reg.py
input-
output-

6. Esemble_blend_reg.py
input-
output-

7. GradientBoostingRegressor.py
input-
output-

8. KNN.py- predict model using KNeighborsClassifier
input- protein wise CC, ECC, functinal activity score and MEM-FET score.
output- predicting the output accuracy and mean_squared_error value.

9. Normalize.py - Scaled the values between 0 to 1.
input- protein wise CC, ECC, functinal activity score and MEM-FET score.
output-Normalized the values of the properties.

10. Sub_cell.py  - Using to calculate subcellular Membership value.
input- Raw file of Sucellular localization (Sub_cellular.txt)
output- Protein wise membership values. Different subcellular files.

11. Get_ess_data.py - Construction of essential dataset.
input- Calculated centrality scores (Centrlity_Ess.csv) and Physicochemical properties score (Physico_chemical_ess.csv). 
output-find protein wise essential dataset (Ess_data.csv)

12. Mem_Cal.py - Membership values of the essential properties.
input- All centrality measures (AllCentrality.xlsx) and All Physicochemical properties ( AllPhysico_chemical.xls)
output-Calculated centrality scores (Centrlity_Ess.csv) and Physicochemical properties score (Physico_chemical_ess.csv).

13. Mem_cen_Sub.py- membership based (alpha, beta, chi, gamma) essential protein prediction based on selected centrality and physico-chemical values.
input- Classifier based calculated centrality properties (Centrlity_Ess.csv) and Physicochemical properties score (Physico_chemical_ess.csv).
output-Essential proteins using membership (alpha, beta, chi, gamma) values.

14.kmeans.py- perform clustering using K-Means clustering.
input- Essential dataset(Ess_data.csv)
output- Clusterwise protein list.

15.Cluster_index.py- Cluster results are validated using validity indices
input-Obtain cluster results from essential dataset.
output- values of the respective validity indices.

16. Sim_loc_score.py- Calculate similarity localized score protein wise.
input-Selected clusters depending on cluster validity indices.
output-protein wise similarity localized score.

17.Thresh_Sim_loc.py- Essential protein prediction using 3-sigma approach
input-protein wise similarity localized score (sim_loc).
output-Three level of threshold(low, mid, high). Essential protein list using thresholds and sim_loc values.

18.level1-2.py,level 1-2_threshold.py , Tot_Socore_levels.py- Find  level-1  proteins of essential proteins using sim_loc values.
input-Protein datasets. sim_loc score of the level-1 proteins of the predicted essential list.
output- predict essential proteins of level-1 of essential list.

19.Ortholog_netComp.py, Fun_subcel_Ortho_list.py- Calculate orthologous value of a protein complex.Listing Ortho_sim_loc properties.
input- Orthologous database, protein database, protein complex.
output- Protein wise orthologous score.

20. Ortho_remove_null.py - Remove proteins without ortholous score.
input- Proteins with orthologous score.
output- Eliminate proteins without orthologous score.

21.Clustering.py- Perform clustering using KMeans, AgglomerativeClustering.
input-Essential dataset(Ess_data.csv).
output-Clusterwise protein list with clustering indices.

22. Physico_chemical.py -Calculate protein wise physico-chemical properties.
input-Protein sequence database.
output-values of different physico-chemical properties values.





