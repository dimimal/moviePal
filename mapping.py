import pandas as pd
import numpy as np

index=[]
data=pd.read_csv("movie_metadata.csv")

actors1=data.loc[:,['actor_1_name']]
actors2=data.loc[:,['actor_2_name']]
actors3=data.loc[:,['actor_3_name']]

directorName=data.loc[:,['director_name']]

actors1=actors1.rename(columns={'actor_1_name' :'actors'})
actors2=actors2.rename(columns={'actor_2_name' :'actors'})
actors3=actors3.rename(columns={'actor_3_name' :'actors'})

actors=[actors1,actors2,actors3]
#actors=actors.as_matrix()
#actors=actors.flatten()
#actors=pd.actors()
actors=pd.concat(actors,ignore_index=True)
#ctors=actors.reset_index(drop=True)
actors=actors.drop_duplicates()
directorName=directorName.drop_duplicates()

sum=pd.concat([actors,directorName],axis=1)

sum.to_csv('actors.csv')

#actors=actors.reshape(len(actors)*3,1)
#saveFile=open('actors.csv','w')
#for i in range(len(actors)):

#	saveFile.write(str(actors[i])+ 'i'+ '\n')	
		
#saveFile.close()


#print actors.index

