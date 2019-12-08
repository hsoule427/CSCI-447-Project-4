import get_dbs as GDB
import main

# This is the file used to run batches of the program for large amounts of data for comparing scheduling algorithms.

# Runs the compiled C code.
from subprocess import call
import subprocess
import xlwt
from xlwt import Workbook

def get_int_input(query):
    val = None
    while True:
        
        try:
            val = int(input(query))
            break
        except:
            print("Invalid input. Please enter an integer.")
       
    return val

lower_bound = 0#get_int_input("Enter lower bound for d range:")
upper_bound = 10#get_int_input("Enter upper bound for d range:")

db_list = GDB.get_db()
fifo_att = None


# Create work book
wb = Workbook()
# Add a sheet to the excel file
sheet = wb.add_sheet('Output')

# Will create sheets for each database
for iter in range(0, len(db_list)):
    sheet.write(0, iter, db_list[iter])

# iter = 1
# for d in range(lower_bound, upper_bound):

#     # Sets up v as a quarter of d.
#     v = 100
#     print("D: ",d)
    # Option to run with seed, or have them be randomized.

for database in db_list:
    print("Running for db:", database)
    
    db_best_result = None
    
    # Range of learning rates to try
    for x in range(2,20,1):
        learning_rate = float(x/10)
        # TODO Tune this per dataset
        db_run_result = main.main(database, learning_rate)
        
        if db_best_result is None or db_run_result < db_best_result:
            db_best_result = db_run_result
            
        print("FIFO RESULT:")
        print(db_run_result)
    
    # TODO: save this to output file
    print("BEST RESULT:", db_best_result, "FOR:", database)
#     # output to excel with each as a unique column
#     sheet.write(iter, 0, d)
#     sheet.write(iter, 1, fifo_att)
#     sheet.write(iter, 2, sjf_att)
#     sheet.write(iter, 3, srt_att)
    
#     iter+=1
    
#     if sjf_att != srt_att:
#         print("Pre-emption!!!")
#     #print("\nRESULT:\nFIFO:",fifo_att,"\nSJF:",sjf_att,"\nSRT:",srt_att,"\n")
    
# wb.save("N:"+ str(n) + "K:"+ str(k) +"d:"+str(lower_bound)+"-"+str(upper_bound)+"-output.xls")