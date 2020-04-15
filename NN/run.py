from lib.coin_center import *
import pstats, cProfile

cProfile.runctx('top_unwrapped, bot_unwrapped = parse_dataset_element(25)', globals(), locals(), 'Profile.prof')
s = pstats.Stats('Profile.prof')
s.strip_dirs().sort_stats('time').print_stats()

save(top_unwrapped, 'top_unwrapped.png')
save(bot_unwrapped, 'bot_unwrapped.png')
