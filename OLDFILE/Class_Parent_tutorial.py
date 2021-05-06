'''super() 函数是用于调用父类(超类)的一个方法。
super 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用等种种问题。
MRO 就是类的方法解析顺序表, 其实也就是继承父类方法时的顺序表'''

class Parent(object):
    def __init__(self):
        self.parent = 'I\'m the parent.'
        print('1-Parent')

    def bar(self, message):
        print("2-{:s} from Parent".format(message))


class Child(Parent):
    def __init__(self):
        # super(Child,self) 首先找到 Child 的父类（就是类 Parent），然后把类 Child 的对象转换为类 Parent 的对象
        super(Child, self).__init__()
        print('3-Child')

    def bar(self, message):
        super(Child, self).bar(message)
        print('4-Child bar fuction')
        print('5-',self.parent)


if __name__ == '__main__':
    fooChild = Child()
    fooChild.bar('HelloWorld')