"""
Tentative train/valid/test split for different tasks
"""

# List: [ID]
odom_sample = [['boreas-2021-09-02-11-42']]

odom_train = [['boreas-2020-12-01-13-26'],
              ['boreas-2020-12-18-13-44'],
              ['boreas-2021-01-15-12-17'],
              ['boreas-2021-01-26-11-22'],
              ['boreas-2021-02-02-14-07'],
              ['boreas-2021-03-02-13-38'],
              ['boreas-2021-03-23-12-43'],
              ['boreas-2021-03-30-14-23'],
              ['boreas-2021-04-13-14-49'],
              ['boreas-2021-04-15-18-55'],
              ['boreas-2021-04-20-14-11'],
              ['boreas-2021-04-29-15-55'],
              ['boreas-2021-05-06-13-19'],
              ['boreas-2021-05-13-16-11'],
              ['boreas-2021-06-03-16-00'],
              ['boreas-2021-06-17-17-52'],
              ['boreas-2021-07-20-17-33'],
              ['boreas-2021-07-27-14-43'],
              ['boreas-2021-08-05-13-34'],
              ['boreas-2021-09-02-11-42'],
              ['boreas-2021-09-09-15-28'], # St George
              ['boreas-2021-09-14-20-00'],
              ['boreas-2021-09-28-19-25'],
              ['boreas-2021-10-15-12-35']]

odom_valid = [['boreas-2021-01-19-15-08'],
              ['boreas-2021-04-08-12-44'],
              ['boreas-2021-09-07-09-35']]

odom_test = [['boreas-2020-12-04-14-00'],
             ['boreas-2021-01-26-10-59'],
             ['boreas-2021-02-09-12-55'],
             ['boreas-2021-03-09-14-23'],
             ['boreas-2021-04-22-15-00'],
             ['boreas-2021-06-29-18-53'],
             ['boreas-2021-06-29-20-43'],
             ['boreas-2021-09-08-21-00'],
             ['boreas-2021-10-05-15-35']] # Night time

# List: [ID, start_ts, end_ts]
obj_sample = []

obj_train = []

obj_valid = []

obj_test = []