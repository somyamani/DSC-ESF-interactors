import sys, os, time, pickle, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time



df_prol = pd.read_csv('/Users/somya/Downloads/Dans_cell_interaction_data/Hs_Proliferative_interactions_rev6.17-orthoSAMap_2.SubType_sig.csv')
df_sec = pd.read_csv('/Users/somya/Downloads/Dans_cell_interaction_data/Hs_Secretory_interactions_rev6.17-orthoSAMap_2.SubType_sig.csv')


# test - are lig-rec interaction matrices the same across prol and sec?
# if not, use lig-rec pairs that are present in both datasets



def get_ligrec_mat():
    t1 = time.time()
    liglist_prol = set(df_prol['ligand_complex'])
    reclist_prol = set(df_prol['receptor_complex'])
    
    liglist_sec = set(df_sec['ligand_complex'])
    reclist_sec = set(df_sec['receptor_complex'])

    liglist = liglist_prol | liglist_sec
    reclist = reclist_prol | reclist_sec

    intmat = np.zeros((len(liglist),len(reclist),2))
    for i0,l in enumerate(liglist):
        print('ligand=',l, 'at time',(time.time()-t1)/60,'min')
        for i1,r in enumerate(reclist):
            intmat[i0,i1,0] = sum((df_prol['ligand_complex']==l)&(df_prol['receptor_complex']==r))
            intmat[i0,i1,1] = sum((df_sec['ligand_complex']==l)&(df_sec['receptor_complex']==r))# intmat[i,j] = how many cell-pairs interact via this interaction
    # specific interaction: very few cell-pairs interact via this interaction + this lig/rec interacts with very few rec/lig
    t2 = time.time()
    print('total time to get interaction matrix:',(t2-t1)/60,'min')
    np.save('ligand_receptor_interaction_matrix.npy',intmat)
    ligreclists = {'liglist':liglist,'reclist':reclist}
    pickle.dump(ligreclists,open('ligand_receptor_lists.pkl','wb'))
    return liglist, reclist, intmat


def get_ligrec_exp_mat():
    t1 = time.time()
    cells_prol = set(df_prol['source'])
    cells_sec = set(df_sec['source'])
    liglist_prol = set(df_prol['ligand_complex'])
    reclist_prol = set(df_prol['receptor_complex'])
    liglist_sec = set(df_sec['ligand_complex'])
    reclist_sec = set(df_sec['receptor_complex'])

    liglist = liglist_prol | liglist_sec
    reclist = reclist_prol | reclist_sec
    celllist = cells_prol | cells_sec

    # also store cell-type ligand/receptor expression matrix for each context
    lig_exp = np.zeros((len(liglist),len(celllist),2))
    rec_exp = np.zeros((len(reclist),len(celllist),2))

    for i0,c0 in enumerate(celllist):
        
        for i1,l in enumerate(liglist):
            lig_exp[i1,i0,0] = np.any((df_prol['ligand_complex']==l)&(df_prol['source']==c0))
            lig_exp[i1,i0,1] = np.any((df_sec['ligand_complex']==l)&(df_sec['source']==c0))

        for i1,r in enumerate(reclist):
            rec_exp[i1,i0,0] = np.any((df_prol['receptor_complex']==r)&(df_prol['source']==c0))
            rec_exp[i1,i0,1] = np.any((df_sec['receptor_complex']==r)&(df_sec['source']==c0))
        print(c0,'in',(time.time()-t1)/60,'min')
    np.save('ligand_expression_matrix.npy',lig_exp)
    np.save('rec_expression_matrix.npy',rec_exp)
    pickle.dump(celllist,open('list_of_all_cells.pkl','wb'))
    return lig_exp, rec_exp, celllist


# for each ESF-interactor [proliferative phase] and DSC interactor [secretory phase], make a separate interaction matrix

def get_ESF_DSC_cellwise_interaction_matrices():
    ligreclists = pickle.load(open('/Users/somya/Downloads/Dans_cell_interaction_data/ligand_receptor_lists.pkl','rb'))
    liglist = np.array(list(ligreclists['liglist']))
    reclist = np.array(list(ligreclists['reclist']))

    intmat = np.load('/Users/somya/Downloads/Dans_cell_interaction_data/ligand_receptor_interaction_matrix.npy')

    # remove from the dfs any rows that have rows with ligand/receptors out of the lists liglist and reclist
    ESF_interactors = {}
    # get all rows of df_prol for which ESF is the source
    # identify the set of target cell-types
    ESF_target_cells = list(set(df_prol['target'][df_prol['source']=='ESF']))
    
    ESF_liglist = np.array(list(set(df_prol['ligand_complex'][df_prol['source']=='ESF'])))
    xy, ESF_ligand_ind, y_ind = np.intersect1d(liglist,ESF_liglist,return_indices=True)
    ESF_ligand_diag = np.zeros((len(liglist)))
    ESF_ligand_diag[ESF_ligand_ind] = 1
    ESF_ligand_diag = np.diag(ESF_ligand_diag)

    # for each target cell-type, get the list of receptors it encodes and get the relevant subset of intmat, save it
    for c1 in ESF_target_cells:
        reclist_c1 = np.array(list(set(df_prol['receptor_complex'][df_prol['target']==c1])))
        xy, c1_reclist_ind, y_ind = np.intersect1d(reclist, reclist_c1,return_indices=True)
        c1_rec_diag = np.zeros((len(reclist)))
        c1_rec_diag[c1_reclist_ind] = 1
        c1_rec_diag = np.diag(c1_rec_diag)
        
        ESF_interactors[c1] = (ESF_ligand_diag@intmat@c1_rec_diag>0).astype(int)

    DSC_interactors = {}
    # get all rows of df_sec for which DSC is the source
    # identify the set of target cell-types
    DSC_target_cells = list(set(df_sec['target'][df_sec['source']=='ESF']))
    
    DSC_liglist = np.array(list(set(df_sec['ligand_complex'][(df_sec['source']=='DSC_early')|(df_sec['source']=='DSC_mid')|(df_sec['source']=='DSC_late')])))
    xy, DSC_ligand_ind, y_ind = np.intersect1d(liglist,DSC_liglist,return_indices=True)
    DSC_ligand_diag = np.zeros((len(liglist)))
    DSC_ligand_diag[DSC_ligand_ind] = 1
    DSC_ligand_diag = np.diag(DSC_ligand_diag)
    # for each target cell-type, get the list of receptors it encodes and get the relevant subset of intmat, save it
    for c1 in DSC_target_cells:
        reclist_c1 = np.array(list(set(df_sec['receptor_complex'][df_sec['target']==c1])))
        xy, c1_reclist_ind, y_ind = np.intersect1d(reclist, reclist_c1,return_indices=True)
        c1_rec_diag = np.zeros((len(reclist)))
        c1_rec_diag[c1_reclist_ind] = 1
        c1_rec_diag = np.diag(c1_rec_diag)
        
        DSC_interactors[c1] = (DSC_ligand_diag@intmat@c1_rec_diag>0).astype(int)

    pickle.dump(ESF_interactors,open('ESF_target_interaction_matrices.pkl','wb'))
    pickle.dump(DSC_interactors,open('DSC_target_interaction_matrices.pkl','wb'))
    return ESF_interactors, DSC_interactors

def compare_ESF_DSC_interaction():
    esf_intmat = pickle.load(open('ESF_target_interaction_matrices.pkl','rb'))
    dsc_intmat = pickle.load(open('DSC_target_interaction_matrices.pkl','rb'))
    
    only_esf_interactors = set(esf_intmat.keys()) - set(dsc_intmat.keys())
    only_dsc_interactors = set(dsc_intmat.keys()) - set(esf_intmat.keys())
    
    common_interactors = set.intersection(set(dsc_intmat.keys()), set(esf_intmat.keys()))
    # identify differential interactions across cell-types
    interaction_diff = {}
    for int1 in common_interactors:
        interaction_diff[int1] = dsc_intmat[int1] - esf_intmat[int1]
    return only_esf_interactors, only_dsc_interactors, interaction_diff


# Want -- interactors of ESF in proliferative phase such that cells differentiating from these ESF-interactors interact with DSC in the secretory phase AND their interactions matrices are distinct.


