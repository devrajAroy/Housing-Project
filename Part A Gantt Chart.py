# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:28:53 2024

@author: Megan
"""

#importing necessary libraries/modules
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#creating lists for tasks, dates and durations, and persons assigned to task
tasks = ['Task allocation', 'Gantt Chart', 'PA 1-2', 'PA 3-4', 'PA 5-6', 'EA', 'Implementation 1', 'Implementation 2', 'Introduction', 'Part B write-up', 'Part C write-up', 'Research', 'Conclusion', 'Intro slides', 'Group practice', 'Finish slides', 'Final group practice']
start_dates = ['2024-04-16', '2024-04-18','2024-04-18', '2024-04-21', '2024-04-24', '2024-04-18', '2024-04-18', '2024-04-18', '2024-04-28', '2024-04-28', '2024-04-27', '2024-04-24', '2024-05-06', '2024-04-20', '2024-04-26', '2024-04-28', '2024-05-08']
end_dates = ['2024-04-18', '2024-04-20', '2024-04-21', '2024-04-24', '2024-04-28', '2024-04-28', '2024-04-26', '2024-04-26', '2024-05-03', '2024-05-06', '2024-05-06', '2024-05-06', '2024-05-09', '2024-04-26', '2024-05-02', '2024-05-08', '2024-05-10',]
person = ['All', 'Megan', 'Vincent and Ismail', 'Vincent and Ismail', 'Megan and Ismail', 'Ismail', 'Rachel', 'Dev', 'Megan and Vincent', 'Rachel and Ismail', 'Rachel and Dev', 'Megan, Rachel and Vincent', 'Vincent and Ismail', 'Megan', 'All', 'All', 'All']

#creating a dataframe from the lists
data = {'Tasks': tasks, 'Start Dates': start_dates, 'End Dates': end_dates, 'Person': person}
df = pd.DataFrame(data)

#converting to datetime and calculating duration
df['Start Dates'] = pd.to_datetime(df['Start Dates'])
df['End Dates'] = pd.to_datetime(df['End Dates'])
df['Days Duration'] = df['End Dates'] - df['Start Dates'] 

#sorting values in chronological order
df = df.sort_values(by = 'Start Dates', ascending = True, ignore_index = True)

#printing the dataframe to visualise the tasks in a table
print(df)

#creating a subplot
fig, ax = plt.subplots()
ax.xaxis_date()

#assigning colours to people
person_colours = {'All': 'red', 'Megan': 'darkorange', 'Vincent and Ismail': 'gold', 'Megan and Ismail': 'lawngreen', 'Ismail': 'forestgreen', 'Rachel': 'lightseagreen', 'Dev': 'deepskyblue', 'Megan and Vincent': 'cornflowerblue', 'Rachel and Ismail': 'mediumblue', 'Rachel and Dev': 'darkorchid', 'Megan, Rachel and Vincent': 'magenta'}
patches = []

#creating legend patches
for person in person_colours:
    patches.append(mpatches.Patch(color = person_colours[person]))
    
#plotting bars, using duration as width
for index, row in df.iterrows():
    plt.barh(y = row['Tasks'], width = row['Days Duration'], left = row['Start Dates'], color = person_colours[row['Person']], alpha = 0.5)
    
#setting title and labels
ax.set_title('Project Plan Gantt Chart (Group Alpha)')
ax.set_xlabel('Date')
ax.set_ylabel('Task')
ax.set_xlim(df['Start Dates'].min(), df['End Dates'].max())
    
#displaying tasks from top to bottom by inverting axis
fig.gca().invert_yaxis()

#defining x axis ticks every 2 days
date_range = pd.date_range(start = df['Start Dates'].min(), end = pd.to_datetime("2024-05-10"), freq = "2D")
ax.set_xticks(date_range)

#rotating labels and adding grid lines
ax.tick_params(axis = 'x', labelrotation = 45)
ax.xaxis.grid(True, alpha = 0.5)

#creat