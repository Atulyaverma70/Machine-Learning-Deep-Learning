## pandas
    * Pandas is a popular open-source Python library used for data manipulation and data analysis.
    * by using pandas you can do postmortem of data.
## creating Dataframe
    ```python
    # using list
    student_data=[
        [100,10,5],
        [85,50,3],
        [40,39,8]
    ]

    pd.DataFrame(student_data,columns=['iq','marks','package'])

    #using dict
    student_dict={
        'iq':[100,50,80],
        'marks':[40,50,30],
        'package':[4,5,6]
    }
    pd.DataFrame(student_dict)

## DataFrame Attributes and methods
    ```python
    # shape
    dp.shape

    #dtypes
    dp.dtypes

    #index
    dp.index

    #columns
    dp.columns

    #values
    dp.values

    #head and tail
    dp.head()
    dp.tail()

    #sample(select randomly columns)
    dp.sample()

    #info
    dp.info()

    #describe(gives the mathmatical summary of data(only for numerical column) like count, mean, std,min,max)
    dp.describe()

    #isnull
    dp.isnull().sum()

    #duplicated
    dp.duplicated()

    #rename(changes the column name)
    dp.rename(columns={'marks':'percentage'},inplace=True)# when we do inplace=true then it makes permanent change

## math methods
    ```python
    #sum
    dp.sum()
        # axis=1 for row wise sum
        # axis=0 for column wise sum
    #mean
    #min,max,median,std,var

## selecting columns from datafram
    ```python
    dp[['student','price','option']]

## selecting row from dataFrame
    ```python
    # iloc: searching using index positions
    # loc: searching using labels

    dp.iloc[0] # it first index row
    dp.iloc[0:5] # it gives the rows form 0 index to 4th index
    dp.iloc[[0,4,5]] # it gives the 0th,4th,5th row

    dp.loc['atulya'] # gives the atulya all data

## selecting both rows and column
    ```python
    dp.iloc[0:3,0:3] # it gives first three row with first three columns

## filtering a dataframe
    ```python
    # like i have ipl data from that i have to perform these operations
    # find all the final winners
    mask=ipl['MatchNumber']=='Final'
    new_df=ipl[mask]
    new_df[['season','WiningTeam']]

    # how many superover finishes have occured
    ipl[ipl['superOver']=='Y'].shape

    # how many matches has srh won in hyderabad
    ipl[(ipl['city']=='hyderabad') & (ipl['winningTeam']=='SRH')].shape


## removing missing values
    ```python
    dp.dropna(inplace=true )

## changing the data type
    ```python 
    #astype
    ipl['id']=ipl['id'].astype('int32')

## important things in pandas
    ```python 
    #between
    dp.between(20,50) # it gives the range of values from 20 to 50
    #clip(it make the all values in the range like we gives the range(20,30) in that all the values that are less then 20 it makes 20 and all the values that are greater then 50 it makes 50)

    # drop_duplicates
    .drop_duplicates() # it drops the all values that are duplicate

    #duplicated (it is used to check duplicated value)

    #isnull(used for checking missing value)

    #dropna(used to all missing values)

    #fillna(used to filll missing values)
    dp.fillna(0)

    #isin()=>used to check at exact value
    dp.isin([49,99])

    #apply(used to apply custom logic)

    #copy(used to create the copy)



