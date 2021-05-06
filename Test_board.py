import tensorflow as tf

A=tf.ones([1,2,3,4,5])
B=tf.ones([4,3,2,1])
C=A.shape[:3].concatenate(B.shape[-2:])
# print(C)


def test(a,*args,**kwargs):
    print(a)
    print(args)
    print(kwargs)
    x=kwargs.get('e')
    print(x)
test(1,2,3,4,b=5, c=6, d=7)