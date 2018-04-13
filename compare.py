import testmixmod
import testlstm
import testgru

def print_score(mod):
    print('Test score:', mod.score)
    print('Test accuracy:', mod.acc)

print("Final scoring:")
print_score(testmixmod)
print_score(testlstm)
print_score(testgru)
