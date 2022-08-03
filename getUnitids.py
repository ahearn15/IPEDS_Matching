import os
import argparse
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct  # Leading Juice for us
import time

parser = argparse.ArgumentParser(description="Script for retrieving IPEDS Unit IDs.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--file', help='[required] Name of file to match institutions with UnitIDs. Must end in .csv or .xlsx.')
parser.add_argument('-o', '--output', help='[optional] Name of file the data is being saved into.', default='matched_data.csv')
parser.add_argument('-i', '--instColumn', help='[optional] Name of column that holds `Institution` variable. If `None`, program will try to find it.', default=None)
parser.add_argument('-st', '--stateColumn', help='[optional] Name of column that holds `State` variable. If `None`, program will try to find it.', default=None)
parser.add_argument('-sf', '--stateFlag', help='[optional] Whether or not input file has `state` variable included.', default=False)
parser.add_argument('-e', '--errorFile', help='[optional] Name of file where non-matched institutions are saved into.', default='non_matched.csv')
parser.add_argument('-v', '--verbose', help='[optional] Verbosity of output.', default=True)

args = parser.parse_args()

def ngrams(string, n=3):
    string = (re.sub(r'[,-./]|\sBD',r'', string)).upper().strip()
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
            M, N, np.asarray(A.indptr, dtype=idx_dtype),
            np.asarray(A.indices, dtype=idx_dtype),
            A.data,
            np.asarray(B.indptr, dtype=idx_dtype),
            np.asarray(B.indices, dtype=idx_dtype),
            B.data,
            ntop,
            lower_bound,
            indptr, indices, data)  
    return(csr_matrix((data,indices,indptr),shape=(M,N)))

def get_matches_df(sparse_matrix, A, B, unitids, states, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

        
    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    unitid = np.empty([nr_matches], dtype=object)
    state = np.empty([nr_matches], dtype=object)
    similarity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = A[sparserows[index]]
        right_side[index] = B[sparsecols[index]]
        state[index] = states[sparsecols[index]]
        similarity[index] = sparse_matrix.data[index]
        unitid[index] = unitids[sparsecols[index]]   
        pct_match = f"{similarity[index]:.1%}"
        if args.verbose == True:
            time.sleep(.0005)
            print('Line ' + str(index + 1) + ': ' + left_side[index] + ' (' + pct_match + ' match)')
    df = pd.DataFrame({'orig': left_side,
                         'new': right_side,
                         'unitid' : unitid,
                         'state' : state,
                         'similarity' : similarity})
    
    return(df)
    
def getUnitIDs(dta):
    truth = pd.read_json('https://raw.githubusercontent.com/ahearn15/IPEDS_Matching/main/ipeds.json').T
    stateFlag = args.stateFlag
    if args.instColumn == None:
        # get column where institution lives
        inst_scores = []
        for col in dta.columns:
            inst_scores.append \
                (((dta[col].astype(str).str.upper().str.contains('COLLEGE')) | \
                  (dta[col].astype(str).str.upper().str.contains('UNIVERSITY')) | \
                  (dta[col].astype(str).str.upper().str.contains('INSTITUTE')) | \
                  (dta[col].astype(str).str.upper().str.contains('SCHOOL')) | \
                  (dta[col].astype(str).str.upper().str.contains('CONSERVATORY')) | \
                  (dta[col].astype(str).str.upper().str.contains('SEMINARY'))).mean())

        inst_col = inst_scores.index(max(inst_scores))
    else:
        inst_col = args.instColumn

    if args.stateColumn == None:
        state_scores = []
        for col in dta.columns:
            state_scores.append \
              (((dta[col].astype(str).str.upper().str.contains('AL')) |
                (dta[col].astype(str).str.upper().str.contains('AK')) |
                (dta[col].astype(str).str.upper().str.contains('AS')) |
                (dta[col].astype(str).str.upper().str.contains('AZ')) |
                (dta[col].astype(str).str.upper().str.contains('AR')) |
                (dta[col].astype(str).str.upper().str.contains('CA')) |
                (dta[col].astype(str).str.upper().str.contains('CO')) |
                (dta[col].astype(str).str.upper().str.contains('CT')) |
                (dta[col].astype(str).str.upper().str.contains('DE')) |
                (dta[col].astype(str).str.upper().str.contains('DC')) |
                (dta[col].astype(str).str.upper().str.contains('FM')) |
                (dta[col].astype(str).str.upper().str.contains('FL')) |
                (dta[col].astype(str).str.upper().str.contains('GA')) |
                (dta[col].astype(str).str.upper().str.contains('GU')) |
                (dta[col].astype(str).str.upper().str.contains('HI')) |
                (dta[col].astype(str).str.upper().str.contains('ID')) |
                (dta[col].astype(str).str.upper().str.contains('IL')) |
                (dta[col].astype(str).str.upper().str.contains('IN')) |
                (dta[col].astype(str).str.upper().str.contains('IA')) |
                (dta[col].astype(str).str.upper().str.contains('KS')) |
                (dta[col].astype(str).str.upper().str.contains('KY')) |
                (dta[col].astype(str).str.upper().str.contains('LA')) |
                (dta[col].astype(str).str.upper().str.contains('ME')) |
                (dta[col].astype(str).str.upper().str.contains('MH')) |
                (dta[col].astype(str).str.upper().str.contains('MD')) |
                (dta[col].astype(str).str.upper().str.contains('MA')) |
                (dta[col].astype(str).str.upper().str.contains('MI')) |
                (dta[col].astype(str).str.upper().str.contains('MN')) |
                (dta[col].astype(str).str.upper().str.contains('MS')) |
                (dta[col].astype(str).str.upper().str.contains('MO')) |
                (dta[col].astype(str).str.upper().str.contains('MT')) |
                (dta[col].astype(str).str.upper().str.contains('NE')) |
                (dta[col].astype(str).str.upper().str.contains('NV')) |
                (dta[col].astype(str).str.upper().str.contains('NH')) |
                (dta[col].astype(str).str.upper().str.contains('NJ')) |
                (dta[col].astype(str).str.upper().str.contains('NM')) |
                (dta[col].astype(str).str.upper().str.contains('NY')) |
                (dta[col].astype(str).str.upper().str.contains('NC')) |
                (dta[col].astype(str).str.upper().str.contains('ND')) |
                (dta[col].astype(str).str.upper().str.contains('MP')) |
                (dta[col].astype(str).str.upper().str.contains('OH')) |
                (dta[col].astype(str).str.upper().str.contains('OK')) |
                (dta[col].astype(str).str.upper().str.contains('OR')) |
                (dta[col].astype(str).str.upper().str.contains('PW')) |
                (dta[col].astype(str).str.upper().str.contains('PA')) |
                (dta[col].astype(str).str.upper().str.contains('PR')) |
                (dta[col].astype(str).str.upper().str.contains('RI')) |
                (dta[col].astype(str).str.upper().str.contains('SC')) |
                (dta[col].astype(str).str.upper().str.contains('SD')) |
                (dta[col].astype(str).str.upper().str.contains('TN')) |
                (dta[col].astype(str).str.upper().str.contains('TX')) |
                (dta[col].astype(str).str.upper().str.contains('UT')) |
                (dta[col].astype(str).str.upper().str.contains('VT')) |
                (dta[col].astype(str).str.upper().str.contains('VI')) |
                (dta[col].astype(str).str.upper().str.contains('VA')) |
                (dta[col].astype(str).str.upper().str.contains('WA')) |
                (dta[col].astype(str).str.upper().str.contains('WV')) |
                (dta[col].astype(str).str.upper().str.contains('WI')) |
                (dta[col].astype(str).str.upper().str.contains('WY')) |
                (dta[col].astype(str).str.upper().str.contains('ALABAMA')) |
                (dta[col].astype(str).str.upper().str.contains('ALASKA')) |
                (dta[col].astype(str).str.upper().str.contains('AMERICAN SAMOA')) |
                (dta[col].astype(str).str.upper().str.contains('ARIZONA')) |
                (dta[col].astype(str).str.upper().str.contains('ARKANSAS')) |
                (dta[col].astype(str).str.upper().str.contains('CALIFORNIA')) |
                (dta[col].astype(str).str.upper().str.contains('COLORADO')) |
                (dta[col].astype(str).str.upper().str.contains('CONNECTICUT')) |
                (dta[col].astype(str).str.upper().str.contains('DELAWARE')) |
                (dta[col].astype(str).str.upper().str.contains('DISTRICT OF COLUMBIA')) |
                (dta[col].astype(str).str.upper().str.contains('FEDERATED STATES OF MICRONESIA')) |
                (dta[col].astype(str).str.upper().str.contains('FLORIDA')) |
                (dta[col].astype(str).str.upper().str.contains('GEORGIA')) |
                (dta[col].astype(str).str.upper().str.contains('GUAM')) |
                (dta[col].astype(str).str.upper().str.contains('HAWAII')) |
                (dta[col].astype(str).str.upper().str.contains('IDAHO')) |
                (dta[col].astype(str).str.upper().str.contains('ILLINOIS')) |
                (dta[col].astype(str).str.upper().str.contains('INDIANA')) |
                (dta[col].astype(str).str.upper().str.contains('IOWA')) |
                (dta[col].astype(str).str.upper().str.contains('KANSAS')) |
                (dta[col].astype(str).str.upper().str.contains('KENTUCKY')) |
                (dta[col].astype(str).str.upper().str.contains('LOUISIANA')) |
                (dta[col].astype(str).str.upper().str.contains('MAINE')) |
                (dta[col].astype(str).str.upper().str.contains('MARSHALL ISLANDS')) |
                (dta[col].astype(str).str.upper().str.contains('MARYLAND')) |
                (dta[col].astype(str).str.upper().str.contains('MASSACHUSETTS')) |
                (dta[col].astype(str).str.upper().str.contains('MICHIGAN')) |
                (dta[col].astype(str).str.upper().str.contains('MINNESOTA')) |
                (dta[col].astype(str).str.upper().str.contains('MISSISSIPPI')) |
                (dta[col].astype(str).str.upper().str.contains('MISSOURI')) |
                (dta[col].astype(str).str.upper().str.contains('MONTANA')) |
                (dta[col].astype(str).str.upper().str.contains('NEBRASKA')) |
                (dta[col].astype(str).str.upper().str.contains('NEVADA')) |
                (dta[col].astype(str).str.upper().str.contains('NEW HAMPSHIRE')) |
                (dta[col].astype(str).str.upper().str.contains('NEW JERSEY')) |
                (dta[col].astype(str).str.upper().str.contains('NEW MEXICO')) |
                (dta[col].astype(str).str.upper().str.contains('NEW YORK')) |
                (dta[col].astype(str).str.upper().str.contains('NORTH CAROLINA')) |
                (dta[col].astype(str).str.upper().str.contains('NORTH DAKOTA')) |
                (dta[col].astype(str).str.upper().str.contains('NORTHERN MARIANA ISLANDS')) |
                (dta[col].astype(str).str.upper().str.contains('OHIO')) |
                (dta[col].astype(str).str.upper().str.contains('OKLAHOMA')) |
                (dta[col].astype(str).str.upper().str.contains('OREGON')) |
                (dta[col].astype(str).str.upper().str.contains('PALAU')) |
                (dta[col].astype(str).str.upper().str.contains('PENNSYLVANIA')) |
                (dta[col].astype(str).str.upper().str.contains('PUERTO RICO')) |
                (dta[col].astype(str).str.upper().str.contains('RHODE ISLAND')) |
                (dta[col].astype(str).str.upper().str.contains('SOUTH CAROLINA')) |
                (dta[col].astype(str).str.upper().str.contains('SOUTH DAKOTA')) |
                (dta[col].astype(str).str.upper().str.contains('TENNESSEE')) |
                (dta[col].astype(str).str.upper().str.contains('TEXAS')) |
                (dta[col].astype(str).str.upper().str.contains('UTAH')) |
                (dta[col].astype(str).str.upper().str.contains('VERMONT')) |
                (dta[col].astype(str).str.upper().str.contains('VIRGIN ISLANDS')) |
                (dta[col].astype(str).str.upper().str.contains('VIRGINIA')) |
                (dta[col].astype(str).str.upper().str.contains('WASHINGTON')) |
                (dta[col].astype(str).str.upper().str.contains('WEST VIRGINIA')) |
                (dta[col].astype(str).str.upper().str.contains('WISCONSIN')) |
                (dta[col].astype(str).str.upper().str.contains('WYOMING'))).mean())
        
        state_col = state_scores.index(max(state_scores))
    else:
        state_scores = [0]
        state_col = args.stateColumn

    # override stateFlag if state column exists
    if max(state_scores) == 1:
        stateFlag = True

    inst_col = dta.columns[inst_col]
    state_col = dta.columns[state_col]

    if (stateFlag == True):
        if (max(dta[state_col].str.len()) <= 2):
            # merge data to get long state
            states = pd.read_json('https://gist.githubusercontent.com/mshafrir/2646763/raw/8b0dbb93521f5d6889502305335104218454c2bf/states_hash.json', orient = 'index').reset_index()
            states.columns = ['state_abbr', 'state_new']
            dta = dta.merge(states, left_on = state_col, right_on = 'state_abbr', how = 'left')

            truth['state'] = truth['state'].fillna(' ')

            truth['institution_long'] = truth['inst_name'] + " " + truth['state']
            dta['inst_col_long'] = dta[inst_col] + " " + dta['state_new']

            insts = list(truth['institution_long'].str.strip())
            dirty_long = list(dta['inst_col_long'].str.strip())

    else:
        insts = list(truth['inst_name'].str.strip())
        dirty_long = list(dta[inst_col].str.strip())

    # replace & with and
    # append st. with saint

    dirty_short = dta[inst_col]
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix_truth = vectorizer.fit_transform(insts)
    tf_idf_matrix_dirty = vectorizer.transform(dirty_long)
    
    matches = awesome_cossim_top(tf_idf_matrix_dirty, tf_idf_matrix_truth.transpose(), 1, 0)

    true_inst_list = list(truth['inst_name'])
    true_unitids = list(truth['unitid'])
    if stateFlag == True:
        true_states = list(truth['state'])
    else:
        true_states = list(truth['state'])

    matches_df = get_matches_df(matches, dirty_short, true_inst_list, true_unitids, true_states, top=0)
    unitids = matches_df['unitid']
    
    non_matches = nonMatches(matches_df)
    dta['UnitID'] = unitids
    first_column = dta.pop('UnitID')
    dta.insert(0, 'UnitID', first_column)
    created_cols = ['state_abbr', 'state_new', 'inst_col_long']
    for col in created_cols:
        if col in dta.columns:
            dta = dta.drop(columns = col)

    return(dta)
    

def nonMatches(dta):
    dta_nm = dta[dta['similarity'].round(10) != 1].sort_values(by = 'similarity')
    if 'state' in dta_nm.columns:
        dta_nm = dta_nm.drop(columns = 'state')
    dta_nm = dta_nm.rename(columns = {'orig' : 'orig_institution',
                                      'new' : 'likely_match',
                                      'unitid' : 'likely_unitid'})
    dta_nm.to_csv(args.error, index = False)

def main():
    try:
        if args.file[-4:] == '.csv':
            dta = pd.read_csv(args.file)
        elif args.file[-5:] == '.xlsx':
            dta = pd.read_excel(args.file)
        elif args.file[-4:] == '.dta':
            dta = pd.read_stata(args.file)   
        else: 
            dta = pd.read_csv(args.file)
        parsed = getUnitIDs(dta)
        parsed.to_csv(args.output, index=False)

    except:
        print('Error loading file. Must be in .csv, .xlsx, or .dta format.')

if __name__ == "__main__":
    main()
