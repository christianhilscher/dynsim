import os
import pandas as pd
import numpy as np

data_path = "/Users/christianhilscher/Desktop/dynsim/src/data_preparation/"
input_path = "/Users/christianhilscher/Desktop/dynsim/input/"

#########################################
def SOEP_to_df(dataf):
    """
    This function takes the SOEP data as a dataframe and returns the the harmonized data such that the rest of the code can work with it. It also renames the columns etc
    """

    dataf = dataf.copy()

    dataf = dataf.rename(columns={'syear': 'year',
                                  'phrf': 'personweight',
                                  'pglabgro': 'gross_earnings',
                                  'hhrf': 'hhweight',
                                  'hgheat': 'heizkosten',
                                  'hgrent': 'bruttokaltmiete',
                                  'pgpsbil': 'education',
                                  'pld0131': 'married1',
                                  'pgtatzeit': 'hours',
                                  'loc1989': 'east1',
                                  'pgemplst': 'employment_status',
                                  'pgfamstd': 'married2',
                                  'pglfs': 'lfs',
                                  'pgerwzeit': 'tenure',
                                  'kal1e01': 'retired'
                                  })

    dataf['orighid'] = dataf['hid']
    # For now motherpid is 0 as a placeholder and maximum age is set to 99
    dataf['motherpid'] = 0
    dataf['age_max'] = 99

    # Generating age
    dataf['age'] = dataf['year'] - dataf['gebjahr']
    dataf['child'] = 0
    dataf.loc[dataf['age']<18,'child']=1

    dataf = _numeric_eduation(dataf)
    dataf = _numeric_employment_status(dataf)
    dataf = _numeric_migration(dataf)
    dataf = _numeric_sex(dataf)
    dataf = _numeric_married(dataf)
    dataf = _numeric_east(dataf)
    dataf = _numeric_couples(dataf)
    dataf = _numeric_bruttokaltmiete(dataf)
    dataf = _numeric_heizkosten(dataf)
    dataf = _numeric_lfs(dataf)
    dataf = _numeric_earnings(dataf)
    dataf = _numeric_hours(dataf)
    dataf = _numeric_working(dataf)
    dataf = make_hh_vars(dataf)

    keep = ['year',
            'pid',
            'hid',
            'personweight',
            'age',
            'gross_earnings',
            'hhweight',
            'heizkosten',
            'bruttokaltmiete',
            'female',
            'east',
            'married',
            'child',
            'in_couple',
            'hours',
            'orighid',
            'motherpid',
            'age_max',
            'education',
            'employment_status',
            'fulltime',
            'lfs',
            'working',
            'tenure',
            'migback',
            'hh_income',
            'hh_youngest_age',
            'n_people',
            'n_children',
            'hh_frac_working',
            'birth',
            'retired']

    dataf = dataf[keep]


    return dataf

def _numeric_eduation(dataf):
    """
    Transforms the string variable education to numeric values
    """
    dataf = dataf.copy()

    dataf.loc[:, "educ"] = 0
    dataf.loc[(dataf['education'] == "[1] Hauptschulabschluss"), "educ"] = 0
    dataf.loc[(dataf['education'] == "[2] Realschulabschluss"), "educ"] = 1
    dataf.loc[(dataf['education'] == "[3] Fachhochschulreife"), "educ"] = 2
    dataf.loc[(dataf['education'] == "[4] Abitur"), "educ"] = 3
    dataf.loc[(dataf['education'] == "[5] Anderer Abschluss"), "educ"] = 4
    dataf.loc[(dataf['education'] == "[6] Ohne Abschluss verlassen"), "educ"] = 5
    dataf.loc[(dataf['education'] == "[7] Noch kein Abschluss"), "educ"] = 6

    dataf.drop("education", axis = 1, inplace = True)
    dataf.rename(columns={'educ': 'education'}, inplace=True)

    return dataf

def _numeric_employment_status(dataf):
    """
    Transforms the string variable employment status to numeric values
    """
    dataf = dataf.copy()

    dataf["emp"] = np.nan
    # Nicht erwerbstätig
    dataf.loc[dataf["age"]<19, "emp"] = 0
    dataf.loc[dataf["employment_status"]=="[5] Nicht erwerbstaetig", "emp"] = 0
    #  Rentner
    dataf.loc[dataf["lfs"]=="[2] NW-age 65 and older", "emp"] = 1
    dataf.loc[dataf["retired"]=="[1] Ja", "emp"] = 1
    # Teilzeit
    empl_list = ["[2] Teilzeitbeschaeftigung", "[3] Ausbildung, Lehre" "[4] Unregelmaessig,geringfuegig erwerbstaet."]
    cond = [i in empl_list for i in dataf["employment_status"]]
    dataf.loc[cond, "emp"] = 2
    # Vollzeit
    dataf.loc[dataf["employment_status"]=="[1] Voll erwerbstaetig", "emp"] = 3
    # Arbeitslos
    #dataf.loc[dataf["employment_status"]=="[6] NW-unemployed", "emp"] = 4



    dataf.drop("employment_status", axis = 1, inplace = True)
    dataf.rename(columns={'emp': 'employment_status'}, inplace=True)

    dataf['fulltime'] = 0
    dataf.loc[dataf['employment_status'] == 3, 'fulltime'] = 1

    return dataf


def _numeric_migration(dataf):
    """
    Transforms the string variable migration background to numeric values
    """
    dataf = dataf.copy()

    dataf['migration'] = 0

    dataf.loc[dataf['migback'] == 0, 'migration'] = 1
    dataf.loc[dataf['migback'] == "[1] kein Migrationshintergrund", 'migration'] = 0

    dataf.drop('migback', axis=1, inplace=True)
    dataf.rename(columns={'migration': 'migback'}, inplace=True)

    return dataf

def _numeric_sex(dataf):
    """
    Transforms the string variable gender to numeric values
    """
    dataf = dataf.copy()

    dataf['female'] = np.NaN

    dataf.loc[dataf['sex'] == "[1] maennlich", 'female'] = 0
    dataf.loc[dataf['sex'] == "[2] weiblich", 'female'] = 1

    dataf.drop('sex', axis=1, inplace=True)

    return dataf

def _numeric_married(dataf):
    """
    Transforms the string variable married to numeric values
    """
    dataf = dataf.copy()

    married_strings = ['[1] Verheiratet, zusammenlebend',
                       '[2] Verheiratet, getrenntlebend',
                       '[6] Eing. gleichg. Partn., zusammenlebend',
                       '[7] Eing. gleichg. Partn., getrenntlebend',
                       '[1] Verheiratet, mit Ehepartner zusammenlebend',
                       '[2] Verheiratet, dauernd getrennt lebend',
                       '[6] Ehepartner im Ausland',
                       '[7] Eingetragene gleichgeschlechtliche Partnerschaft zusammenlebend',
                       '[8] Eingetragene gleichgeschlechtliche Partnerschaft getrennt lebend']

    condition = [status in  married_strings for status in dataf['married2']]

    dataf['married'] = 0

    dataf.loc[condition, 'married'] = 1

    #dataf.drop('married2', axis=1, inplace=True)

    return dataf

def _numeric_east(dataf):
    dataf = dataf.copy()

    neuelaender = ['[13] Mecklenburg-Vorpommern',
                   '[12] Brandenburg',
                   '[14] Sachsen',
                   '[15] Sachsen-Anhalt',
                   '[16] Thueringen']

    notapplicable = ['[-1] keine Angabe',
                     '[-3] nicht valide']

    condition_yes = [status in  neuelaender for status in dataf['bula']]
    condition_no = [status in  notapplicable for status in dataf['bula']]

    dataf['east'] = 0

    dataf.loc[condition_yes, 'east'] = 1
    dataf.loc[condition_no, 'east'] = np.nan


    dataf.drop('bula', axis=1, inplace=True)

    return dataf

def _numeric_couples(dataf):
    dataf = dataf.copy()

    couples = ['[5] 5 Paar + K. GT 16',
               '[4] 4 Paar + K. LE 16',
               '[2] 2 (Ehe-)Paar ohne K.',
               '[6] 6 Paar + K. LE und GT 16']

    condition_yes = [couple in  couples for couple in dataf['hgtyp1hh']]

    dataf['in_couple'] = 0

    dataf.loc[condition_yes, 'in_couple'] = 1


    dataf.drop('hgtyp1hh', axis=1, inplace=True)

    return dataf

def _numeric_lfs(dataf):
    dataf = dataf.copy()

    dataf['lfs_tmp'] = 0

    dataf.loc[dataf['lfs'] == "[11] Working" , 'lfs_tmp'] = 1
    dataf.loc[dataf['lfs'] == "[12] Working but NW past 7 days" , 'lfs_tmp'] = 1

    dataf.drop('lfs', axis=1, inplace=True)
    dataf.rename(columns={'lfs_tmp': 'lfs'}, inplace=True)

    return dataf

def _numeric_working(dataf):
    dataf = dataf.copy()

    dataf['working'] = 0
    dataf['parttime'] = 0
    dataf['fulltime'] = 0

    dataf.loc[(dataf['employment_status'] == 2) & \
              (dataf['employment_status'] == 3) , 'working'] = 1
    dataf.loc[(dataf["lfs"]==1)&(dataf["hours"]>0), "working"] = 1

    dataf.loc[dataf["employment_status"]==0, "working"] = 0
    dataf.loc[dataf["employment_status"]==1, "working"] = 0

    dataf.loc[dataf['employment_status'] == 2, 'parttime'] = 1
    dataf.loc[dataf['employment_status'] == 3, 'fulltime'] = 1

    return dataf

def _numeric_earnings(dataf):
    dataf = dataf.copy()

    condition = [type(typ)==str for typ in dataf['gross_earnings']]
    dataf.loc[condition, 'gross_earnings'] = np.nan

    dataf.loc[(dataf["gross_earnings"].isna()) & (dataf["employment_status"] == 0) & (dataf["lfs"]==0), "gross_earnings"] = 0
    dataf.loc[(dataf["gross_earnings"].isna()) & (dataf["employment_status"] == 1) & (dataf["lfs"]==0), "gross_earnings"] = 0
    dataf['gross_earnings'] = dataf['gross_earnings'].astype(np.float64)

    return dataf

def _numeric_heizkosten(dataf):
    dataf = dataf.copy()

    condition = [type(typ)==str for typ in dataf['heizkosten']]
    dataf.loc[condition, 'heizkosten'] = np.nan

    dataf['heizkosten'] = dataf['heizkosten'].astype(np.float64)

    return dataf

def _numeric_bruttokaltmiete(dataf):
    dataf = dataf.copy()

    condition = [type(typ)==str for typ in dataf['bruttokaltmiete']]
    dataf.loc[condition, 'bruttokaltmiete'] = np.nan

    dataf['bruttokaltmiete'] = dataf['bruttokaltmiete'].astype(np.float64)

    return dataf

def _numeric_hours(dataf):
    dataf = dataf.copy()

    condition = [type(typ)==str for typ in dataf['hours']]
    dataf.loc[condition, 'hours'] = np.nan


    dataf['hours'] = dataf['hours'].astype(np.float64)
    dataf.loc[(dataf["hours"].isna()) & (dataf["employment_status"] == 0) & (dataf["lfs"]==0), "hours"] = 0
    dataf.loc[(dataf["hours"].isna()) & (dataf["employment_status"] == 1) & (dataf["lfs"]==0), "hours"] = 0
    return dataf

# Making household wide variables
def make_hh_vars(dataf):
    """
    Generating variables which belong to one household such as HH-income
    """
    dataf = dataf.copy()
    dataf = _get_multiindex(dataf)

    dataf = _hh_income(dataf)
    dataf = _hh_age_youngest(dataf)
    dataf = _hh_fraction_working(dataf)
    dataf = _indicate_birth(dataf)
    dataf.reset_index(inplace=True, drop=True)
    return dataf


def _get_tupleindices(dataf):
    years = dataf['year'].tolist()
    hids = dataf['hid'].tolist()
    return list(zip(years, hids))

def _get_multiindex(dataf):
    dataf = dataf.copy()
    index_list = _get_tupleindices(dataf)

    mindex = pd.MultiIndex.from_tuples(index_list, names=['year' ,
                                                          'hid'])
    dataf_out = dataf.set_index(mindex)
    dataf_out = dataf_out.sort_index(level=1)

    return dataf_out

def _hh_income(dataf):
    dataf = dataf.copy()
    earnings = dataf.groupby(level=['year', 'hid'])['gross_earnings'].sum()
    dataf['hh_income'] = earnings
    return dataf

def _hh_size(dataf):
    dataf = dataf.copy()
    size = dataf.groupby(level=['year', 'hid'])['gross_earnings'].size()
    dataf['n_people'] = size
    return dataf

def _hh_children(dataf):
    dataf = dataf.copy()
    children = dataf.groupby(level=['year', 'hid'])['child'].sum()
    dataf['n_children'] = children
    return dataf

def _hh_fraction_working(dataf):
    """
    Returns in % the fraction of adults in a HH working
    """
    dataf = dataf.copy()

    dataf = _hh_size(dataf)
    dataf = _hh_children(dataf)

    total = dataf.groupby(level=['year', 'hid'])['working'].sum()
    dataf['total_working'] = total

    dataf['n_adults'] = dataf['n_people'] - dataf['n_children']
    dataf['hh_frac_working'] = dataf['total_working']/dataf['n_adults']
    dataf.loc[dataf['n_adults']==0, 'hh_frac_working'] = 0

    dataf.drop(['total_working', 'n_adults'], axis=1, inplace=True)
    return dataf

def _hh_age_youngest(dataf):
    """
    Returns the age of the youngest person in HH
    Is relevant when it comes to the estimation of the labor force participation of the mothers
    """
    dataf = dataf.copy()

    smallest_age = dataf.groupby(level=['year', 'hid'])['age'].min()
    dataf['hh_youngest_age'] = smallest_age
    return dataf

def _indicate_birth(dataf):
    """
    Indictaes whether a mother has had a baby in that particular year
    """

    dataf = dataf.copy()

    minage = dataf.groupby(level=['year', 'hid'])['age'].min()
    dataf["minage"] = minage
    dataf["birth"] = 0
    dataf.loc[(dataf["minage"]==0)&(dataf["female"]==1)&(dataf["child"]==0), "birth"] = 1
    dataf.drop("minage", axis=1, inplace=True)

    return dataf
##############################################################################
