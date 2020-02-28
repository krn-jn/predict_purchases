# This is main file that should be used to call code from other modules
# You can write code for parsing input parameters but there should not
# be any other logic in this file

import time
from src.predict_purchases import predict_purchases

start_time = time.time()
print('\nExecuting...\n')

if __name__ == "__main__":

    obj = predict_purchases()

    # deprecated
    # obj.read_single_board()
    # obj.read_push_all_boards()
    # obj.sync_databases()

    # running test SQL queries against DB of choice
    # obj.test_stuff()
    
    # main menu
    obj.start_execution()

elapsed_time = time.time() - start_time

print('\nExecution time:', elapsed_time)
print('\n')