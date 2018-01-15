# 1. npという名前でnumpyをimportする
import numpy as np

# 2. numpyのバージョンと構成を表示する
def exe_2():
    print(np.__version__)
    np.show_config()

# 3. サイズ10のヌルベクトルを作成する
def exe_3():
    Z = np.zeros(10)
    print(Z)

# 4. 任意の配列のメモリサイズを見つける方法
def exe_4():
    Z = np.zeros((10,10))
    print("%d bytes" % (Z.size * Z.itemsize))

# 5. コマンドラインからnumpy add関数のドキュメントを取得する方法は？
# %run `python -c "import numpy; numpy.info(numpy.add)"`

# 6. 0を要素とする配列を10用意し、4番目のデータを1に変更する
def exe_6():
    Z = np.zeros(10)
    Z[4] = 1
    print(Z)

# 7. 値の範囲が10から49までのベクトルを作成する
def exe_7():
    Z = np.arange(10,50)
    print(Z)

# 8. ベクトルを反転する（最初の要素が最後になる）
def exe_8():
    Z = np.arange(50)
    Z = Z[::-1]
    print(Z)

# 9. 0〜8の範囲の値を持つ3x3行列を作成する
def exe_9():
    Z = np.arange(9).reshape(3,3)
    print(Z)

# 10. [1,2,0,0,4,0] の中から0ではない要素を見つける
def exe_10():
    nz = np.nonzero([1,2,0,0,4,0])
    print(nz)

# 11. 3x3の単位行列を作成する
def exe_11():
    Z = np.eye(3)
    print(Z)

# 12. ランダムな値を持つ3x3x3配列を作成する
def exe_12():
    Z = np.random.random((3,3,3))
    print(Z)

# 13. ランダムな値を持つ10x10の配列を作成し、最小値と最大値を探します。
def exe_13():
    Z = np.random.random((10,10))
    Zmin, Zmax = Z.min(), Z.max()
    print(Zmin, Zmax)

# 14. サイズ30のランダムベクトルを作成し、平均値を求める
def exe_14():
    Z = np.random.random(30)
    m = Z.mean()
    print(m)

# 15. 境界に1、内部に0を持つ2次元配列を作成する
def exe_15():
    Z = np.ones((10,10))
    Z[1:-1,1:-1] = 0
    print(Z)

# 16. 既存の配列の周りに（0で塗りつぶした）境界線を追加するには？
def exe_16():
    Z = np.ones((5,5))
    Z = np.pad(Z, pad_width=1, mode='constant', constant_values=0)
    print(Z)

# 17. 次の式の結果はどうなりますか？
def exe_17():
    print(0 * np.nan)
    print(np.nan == np.nan)
    print(np.inf > np.nan)
    print(np.nan - np.nan)
    print(0.3 == 3 * 0.1)

# 18. 対角線のすぐ下の1,2,3,4の値を持つ5x5行列を作成する
def exe_18():
    Z = np.diag(1+np.arange(4),k=-1)
    print(Z)

# 19. 8×8行列を作成し、チェッカーボードパターンで記入
def exe_19():
    Z = np.zeros((8,8),dtype=int)
    Z[1::2,::2] = 1
    Z[::2,1::2] = 1
    print(Z)

# 20. 100番目の要素のインデックス（x、y、z）は何ですか？（6,7,8）形状の配列を考えてみましょう。
def exe_20():
    print(np.unravel_index(100,(6,7,8)))

# 21. tile 関数を使用してチェッカーボードの8×8行列を作成する
def exe_21():
    Z= np.tile(np.array([[0,1],[1,0]]), (4,4))
    print(Z)

# 22. 5x5ランダム行列を正規化する
def exe_22():
    Z = np.random.random((5,5))
    Zmax, Zmin = Z.max(), Z.min()
    Z = (Z - Zmin)/(Zmax - Zmin)
    print(Z)

# 23. 色を4つの符号なしバイト（RGBA）として記述するカスタムdtypeを作成します。
def exe_23():
    color = np.dtype([("r", np.ubyte, 1),
        ("g", np.ubyte, 1),
        ("b", np.ubyte, 1),
        ("a", np.ubyte, 1)])

# 24. 5x3行列に3x2行列を乗算する（実際の行列積）
def exe_24():
    Z = np.dot(np.ones((5,3)), np.ones((3,2)))
    print(Z)

    # Python 3.5以降の代替解決策
    Z = np.ones((5,3)) @ np.ones((3,2))
    print(Z)

# 25. 与えられた1Dの配列は、3と8の間のすべての要素に-1をかけます
def exe_25():
    Z = np.arange(11)
    Z[(3 <= Z) & (Z <= 8)] *= -1
    print(Z)

## 26. 次のスクリプトの出力は？
#def exe_26():
#    print(sum(range(5),-1)) # 9
#    from numpy import *
#    print(sum(range(5),-1)) # 10

# 27. 整数ベクトルZを考えてみましょう。
def exe_27():
    Z = 10
    Z**Z         # 10000000000
    2 << Z >> 2  # 512
    Z <- Z       # false
    1j*Z         # 10j
    Z/1/1        # 10.0
    Z<Z>Z        # false

# 28. 次の式の結果はどうなりますか？
def exe_28():
    print(np.array(0) / np.array(0)) # nan
    print(np.array(0) // np.array(0)) # 0
    print(np.array([np.nan]).astype(int).astype(float)) # [-9.22337204e+18]

# 29. 浮動小数点型配列をゼロから丸める方法は？
def exe_29():
     Z = np.random.uniform(-10,+10,10)
     print (np.copysign(np.ceil(np.abs(Z)), Z))

# 30. どのように2つの配列間の共通の値を見つけますか？
def exe_30():
    Z1 = np.random.randint(0,10,10) # [8 2 4 4 2 1 1 7 4 4]
    Z2 = np.random.randint(0,10,10) # [9 6 0 9 7 9 1 6 3 3]
    print(np.intersect1d(Z1,Z2))    # [1 7]

# 31. すべての小さな警告を無視する方法は（非推奨）？
def exe_31():
    # 自殺モードオン
    defaults = np.seterr(all="ignore")
    Z = np.ones(1) / 0
    # 解除
    _ = np.seterr(**defaults)

    # こういうやり方もある。
    with np.errstate(divide='ignore'):
        Z = np.ones(1) / 0

# 32. 次の式はあってますか？
def exe_32():
    p.emath == np.emath.sqrt(-1) # false

# 33. 昨日、今日、そして明日の日付を取得するには？
def exe_33():
    yesterday = np.datetime64('today', 'D') - np.timedelta64(1, 'D')
    today     = np.datetime64('today', 'D')
    tomorrow  = np.datetime64('today', 'D') + np.timedelta64(1, 'D')

# 34. 2016年7月の月に対応するすべての日付を取得するにはどうすればよいですか？
def exe_34():
    Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
    print(Z)

# 35. ((A+B)*(-A/2)) の計算方法（コピーなし）
def exe_35():
    A = np.ones(3)*1
    B = np.ones(3)*2
    C = np.ones(3)*3
    np.add(A,B,out=B)
    np.divide(A,2,out=A)
    np.negative(A,out=A)
    np.multiply(A,B,out=A)

# 36. 5つの異なる方法を使用してランダム配列の整数部分を抽出する
def exe_36():
    Z = np.random.uniform(0,10,10)

    print (Z - Z%1)
    print (np.floor(Z))
    print (np.ceil(Z)-1)
    print (Z.astype(int))
    print (np.trunc(Z))

# 37. 行の値が0から4までの5x5行列を作成する
def exe_37():
    Z = np.zeros((5,5))
    Z += np.arange(5)
    print(Z)

# 38. 10個の整数を生成し、それを使って配列を作成するジェネレータ関数を考えてみましょう
def generate():
    for x in range(10):
        yield x
def exe_38():
    Z = np.fromiter(generate(),dtype=float,count=-1)
    print(Z) # [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]

# 39. 0から1の範囲の値を持つサイズ10のベクトルを作成します。両方とも除外されます
def exe_39():
    Z = np.linspace(0,1,11,endpoint=False)[1:]
    print(Z)

# 40. サイズ10のランダムベクトルを作成してソートする
def exe_40():
    Z = np.random.random(10)
    Z.sort()
    print(Z)

# 41. どのようにすれば小さな配列をnp.sumより速く合計値を導けますか？
def exe_41():
    Z = np.arange(10)
    np.add.reduceat(10)

# 42. 2つのランダムな配列AとBを考え、それらが等しいかどうかをチェックする
def exe_42():
    A = np.random.randint(0,2,5)
    B = np.random.randint(0,2,5)

    # 配列の形状が同じであり、値の比較の許容差があると仮定します
    equal = np.allclose(A,B)
    print(equal)

    # 形状と要素の値の両方をチェックし、公差はありません（値は正確に等しくなければなりません）
    equal = np.array_equal(A,B)
    print(equal)

# 43. 配列を不変（読み取り専用）にする
def exe_43():
    Z = np.zeros(10)
    Z.flags.writeable = False
    Z[0] = 1

# 44. デカルト座標を表すランダムな10x2行列を考え、それらを極座標に変換する
def exe_44():
    Z = np.random.random((10,2))
    X,Y = Z[:,0], Z[:,1]
    R = np.sqrt(X**2+Y**2)
    T = np.arctan2(Y,X)
    print(R)
    print(T)

# 45. サイズ10のランダムベクトルを作成し、最大値を0に置き換えます
def exe_45():
    Z = np.random.random(10)
    Z[Z.argmax()] = 0
    print(Z)

# 46. [0,1] x [0,1]領域をカバーするx座標とy座標を持つ構造化配列を作成する
def exe_46():
    Z = np.zeros((5,5), [('x',float),('y',float)])
    Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                                 np.linspace(0,1,5))
    print(Z)

# 47. 2つの配列XとYが与えられる時、Cauchy matrix C (Cij =1/(xi - yj)) を構築する
def exe_47():
    X = np.arange(8)
    Y = X + 0.5
    C = 1.0 / np.subtract.outer(X, Y)

# 48. 各numpyスカラー型の最小値と最大値を表示する
def exe_48():
    for dtype in [np.int8, np.int32, np.int64]:
        print(np.iinfo(dtype).min)
        print(np.iinfo(dtype).max)

# -128
# 127
# -2147483648
# 2147483647
# -9223372036854775808
# 9223372036854775807

    for dtype in [np.float32, np.float64]:
        print(np.finfo(dtype).min)
        print(np.finfo(dtype).max)
        print(np.finfo(dtype).eps)

# -3.4028235e+38
# 3.4028235e+38
# 1.1920929e-07
# -1.7976931348623157e+308
# 1.7976931348623157e+308
# 2.220446049250313e-16

# 49. どのように配列のすべての値をprintするのですか？
def exe_49():
    np.set_printoptions(threshold=np.nan)
    Z = np.zeros((16,16))
    print(Z)

# 50. ベクトル内で（スカラーに対して）最も近い値を見つけるにはどうすればよいですか？
def exe_50():
    Z = np.arange(100)
    v = np.random.uniform(0,100)
    index = (np.abs(Z-v)).argmin()
    print(Z[index])

# 51. position (x,y) とcolor (r,g,b) を表す構造化配列を作成します。
def exe_51():
    Z = np.zeros(10, [ ('position', [ ('x', float, 1),
                                      ('y', float, 1)]),
                       ('color',    [ ('r', float, 1),
                                      ('g', float, 1),
                                      ('b', float, 1)])])
    print(Z)

# 52. 座標を表す形状（100,2）を有するランダムベクトルを考え、ポイント毎の距離を見つける
def exe_52():
    Z = np.random.random((10,2))
    X,Y = np.atleast_2d(Z[:,0], Z[:,1])
    D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
    print(D)
    
    # Much faster with scipy
    import scipy
    # Thanks Gavin Heverly-Coulson (#issue 1)
    import scipy.spatial
    
    Z = np.random.random((10,2))
    D = scipy.spatial.distance.cdist(Z,Z)
    print(D)

# 53. 浮動小数点（32ビット）配列を整数（32ビット）に変換するには？
def exe_53():
    Z = np.arange(10, dtype=np.int32)
    Z = Z.astype(np.float32, copy=False)
    print(Z)

# 54. 次のファイルを読むには？
def exe_54():
    from io import StringIO

    # Fake file 
    s = StringIO("""1, 2, 3, 4, 5\n
                    6,  ,  , 7, 8\n
                    ,  , 9,10,11\n""")
    Z = np.genfromtxt(s, delimiter=",", dtype=np.int)
    print(Z)

# 55. numpy配列の列挙に相当するものは何ですか？
def exe_55():
    Z = np.arange(9).reshape(3,3)
    for index, value in np.ndenumerate(Z):
        print(index, value)

    for index in np.ndindex(Z.shape):
        print(index, Z[index])

# 56. ジェネリック2Dガウスのような配列を生成する
def exe_56():
    X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
    D = np.sqrt(X*X+Y*Y)
    sigma, mu = 1.0, 0.0
    G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
    print(G)

# 57. p要素をランダムに2次元配列に配置する方法は？
def exe_57():
    n = 10
    p = 3
    Z = np.zeros((n,n))
    np.put(Z, np.random.choice(range(n*n), p, replace=False),1)
    print(Z)

# 58. 行列の各行の平均を引く
def exe_58():
    X = np.random.rand(5, 10)
    Y = X - X.mean(axis=1, keepdims=True)
    print(Y)

# 59. 配列をn番目の列でソートする方法は？
def exe_59():
    Z = np.random.randint(0,10,(3,3))
    print(Z)
    print(Z[Z[:,1].argsort()])

# 60. 与えられた2D配列に空の列があるかどうかを調べるには？
def exe_60():
    Z = np.random.randint(0,3,(3,10))
    print((~Z.any(axis=0)).any())

# 61. 配列内の特定の値から最も近い値を見つける
def exe_61():
    Z = np.random.uniform(0,1,10)
    z = 0.5
    m = Z.flat[np.abs(Z - z).argmin()]
    print(m)

# 62. shape（1,3）と（3,1）を持つ2つの配列を考えると、イテレータを使ってその和を計算する方法は？
def exe_62():
    A = np.arange(3).reshape(3,1)
    B = np.arange(3).reshape(1,3)
    it = np.nditer([A,B,None])
    for x,y,z in it: z[...] = x + y
    print(it.operands[2])
    # [[0 1 2]
    # [1 2 3]
    #  [2 3 4]]

# 63. name属性を持つ配列クラスを作成する
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name")

def exe_63():
    Z = NamedArray(np.arange(10), "range_10")
    print (Z.name)

# 64. 与えられたベクトルを考えて、第2のベクトルによってインデックスされた各要素に1を加える方法（繰り返しインデックスに注意してください）？

def exe_64():
    Z = np.ones(10)
    I = np.random.randint(0,len(Z),20)
    Z += np.bincount(I, minlength=len(Z))
    print(Z)
    
    # Another solution
    np.add.at(Z, I, 1)
    print(Z)

# 65. インデックスリスト（I）に基づいてベクトル（X）の要素を配列（F）に累積する方法は？
def exe_65():
    X = [1,2,3,4,5,6]
    I = [1,3,9,3,4,1]
    F = np.bincount(I,X)
    print(F)

# 66. （dtype = ubyte）の（w、h、3）画像を考慮して、固有色の数を計算する
def exe_66():
    w,h = 16,16
    I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
    F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
    n = len(np.unique(F))
    print(np.unique(I))

# 67. 4次元配列を考えると、一度に最後の2軸を合計する方法は？
def exe_67():
    A = np.random.randint(0,10,(3,4,3,4))
    # solution by passing a tuple of axes (introduced in numpy 1.7.0)
    sum = A.sum(axis=(-2,-1))
    print(sum)
    # solution by flattening the last two dimensions into one
    # (useful for functions that don't accept tuples for axis argument)
    sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
    print(sum)

# 68. 1次元ベクトルDを考えると、サブセットインデックスを記述する同じサイズのベクトルSを使用してDのサブセットの手段をどのように計算するのか？
def exe_68():
    D = np.random.uniform(0,1,100)
    S = np.random.randint(0,10,100)
    D_sums = np.bincount(S, weights=D)
    D_counts = np.bincount(S)
    D_means = D_sums / D_counts
    print(D_means)

# 69. ドットプロダクトの対角をどうやって取得するのですか？
def exe_69():
    A = np.random.uniform(0,1,(5,5))
    B = np.random.uniform(0,1,(5,5))

    # Slow version  
    np.diag(np.dot(A, B))

    # Fast version
    np.sum(A * B.T, axis=1)

    # Faster version
    np.einsum("ij,ji->i", A, B)

# 70. ベクトル[1、2、3、4、5]、各値の間に3つの連続するゼロがインターリーブされた新しいベクトルを構築する方法を考えてみましょう。
def exe_70():
    Z = np.array([1,2,3,4,5])
    nz = 3
    Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
    Z0[::nz+1] = Z
    print(Z0)
