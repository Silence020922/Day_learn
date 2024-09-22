## 算子原型
- 支持int16,int32,half,float四种不同数据类型

- 支持任意维度的输

- 支持如下范围的`Input_x`, `Input_y`输入格式
    - X 与 Y 大小相同
    - Y 为常量
    - Y 大小等于 X 最后一维

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">-</td><td align="center">-</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">-</td><td align="center">-</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">-</td><td align="center">-</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table>
