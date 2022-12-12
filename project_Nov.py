def isPalindromeString(st): # 回文判断函数
    if "str" in str(type(st)):
        text = str(st)
        if len(text) < 2:
            return True
        else:
            if text[0] != text[-1]:
                return False
        return isPalindromeString(text[1:-1])
    else:
        print("Error")

def int_list(): # 将录入数据转化为列表
    global data_list
    data_list = []
    temp = "Y"
    while temp != "N":
        temp = input("please enter int(enter N to exit ) : ")
        try:
            temp = int(temp)
            data_list.append(temp)
        except:
            if temp != "N":
                print('check your input!')

def B_seach(start,end,data): # 二分查找函数，仅接受排序后数据列表
    if "int" in str(type(data)):
        if start <= end:
            mid = (start + end)//2
            if data > data_list[mid]:
                B_seach(mid+1,end,data)
            elif data < data_list[mid]:
                B_seach(start,mid-1,data)
            else:
                print("find it!The subscript:",mid)
        else:
            print("your data is not in data_list")


def main_fun(): # 功能实现函数
    int_list()
    data_list.sort(reverse=True)
    print(data_list)
    n = len(data_list)-1
    try:
        data = int(input("The int you want to find is : "))
        B_seach(0,n,data)
    except:
        print("somthing failed,please try it again ")
        main_fun()

# isPalindromeString()
# main_fun()