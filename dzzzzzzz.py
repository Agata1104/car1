import kotlin.arrayOf
import kotlin.math.*
import kotlin.random.Random

open class Polynome {
    var degree: Int
    var coeffs: DoubleArray

    constructor(coeffs: DoubleArray) {
        this.coeffs = clean(coeffs)
        this.degree = this.coeffs.size - 1
    }
    constructor() : this(doubleArrayOf(0.0))
    constructor(i: Int) : this(DoubleArray(i, { 0.0 })) // Лямбда-выражение

    fun clean(coeffs: DoubleArray): DoubleArray {
        var my_degree = coeffs.size-1
        for (i in my_degree downTo 0) {
            if (coeffs[i] == 0.0) my_degree -= 1
            else break
        }
        my_degree = max(my_degree, 0)
        var my_coeffs = DoubleArray(my_degree+1, {i:Int->coeffs[i]}) // Лямбда-выражение
        /*for(i in 0..my_degree+1){
            my_coeffs[i] = this.coeffs[i]
        }*/
        return my_coeffs
    }

    fun derivative(): Polynome {
        if (this.degree == 0) {
            return Polynome(doubleArrayOf(0.0))
        }

        val resultCoeffs = DoubleArray(this.degree)
        for (i in 0 until this.degree) {
            resultCoeffs[i] = this.coeffs[i + 1] * (i + 1)
        }

        return Polynome(resultCoeffs)
    }


    override fun toString(): String{
        var str=""
        for(i in this.degree downTo 0){
            if(coeffs[i]==0.0) continue
            str+={c:Double->if(c>0) "+" else "-"}(coeffs[i]) // Лямбда-выражение
            if (str=="+") str=""
            str+="${abs(coeffs[i])}"
            if (i!=0) {str+="*x^$i"}
        }
        return str
    }

    operator fun plus(other:Polynome):Polynome{
        var my_degree = max(this.degree, other.degree)
        var my_coeffs: DoubleArray = DoubleArray(my_degree+1, {0.0}) // Лямбда-выражение
        for(i in 0..my_degree){
            var c = {c: DoubleArray , d: Int-> if(d>=i) c[i] else 0.0 }
            val x = c(this.coeffs, this.degree)
            val y = c(other.coeffs, other.degree)
            my_coeffs[i] += x+y
        }
        return Polynome(my_coeffs)
    }

    operator fun times(k: Double):Polynome{
        return Polynome(DoubleArray(this.degree+1, {i:Int->coeffs[i]*k}))
    }

    operator fun Double.times(p: Polynome)=p.times((this))

    operator fun times(other:Polynome):Polynome{
        var my_coeffs = DoubleArray(this.degree+other.degree+1, {0.0})
        for(i in 0..this.degree)
            for( j in 0..other.degree)
                my_coeffs[i+j] += this.coeffs[i]*other.coeffs[j]
        return Polynome(my_coeffs)
    }

    fun valueAt(x: Double): Double {
        var result = 0.0
        for (i in 0..degree) {
            result += coeffs[i] * x.pow(i)
        }
        return result
    }

}

interface Nodes {
    val nodes: List<Double>
    var size: Int
}

data class EquispacedNodes(
    var start: Double,
    var end: Double,
    override var size: Int
) : Nodes {
    override val nodes: List<Double>
        get() {
            if (size <= 2) throw IllegalArgumentException("Должно быть больше 2-х")
            if (start >= end) throw IllegalArgumentException("Start должен быть меньше чем end")

            return List(size) { i -> // Лямбда-выражение для создания списка
                start + i * (end - start) / (size - 1)
            }
        }
}

data class RandomNodes(
    override var size: Int,
    var min: Double = 0.0,
    var max: Double = 2.0
) : Nodes {
    override val nodes: List<Double>
        get() {
            if (size <= 0) throw IllegalArgumentException("size Должно быть больше 0")
            if (max < min) throw IllegalArgumentException("max должен быть больше чем min")

            return List(size) { Random.nextDouble(min, max) }.sorted()
        }
}

data class ChebyshevNodes(
    override var size: Int,
    var a: Double = -1.0,
    var b: Double = 1.0
) : Nodes {
    override val nodes: List<Double>
        get(){
            if (size <= 0) throw IllegalArgumentException("size Должно быть больше 0")
            if (a>b) throw IllegalArgumentException("a>b = error")

            return List(size){i ->
                var chebyshevNode = -cos((2 * i + 1) * PI / (2 * size))
                (b - a) / 2 * chebyshevNode + (a + b) / 2
            }.sorted()
        }
}


class Newton {
    var points: List<Double>
    var values: List<Double>

    private fun check(){
        if (points.size != values.size) throw IllegalArgumentException("Points size != value size")
        if (points.toSet().size != points.size) throw IllegalArgumentException("points not ok")
    }

    constructor(points: List<Double>, values: List<Double>) {
        this.points = points
        this.values = values
        check()
    }

    private fun dividedDifferences(): DoubleArray {
        var sortedPairs = points.zip(values).sortedBy { it.first }
        var sortedPoints = sortedPairs.map { it.first } // Лямбда-выражение
        var sortedValues = sortedPairs.map { it.second } // Лямбда-выражение

        var n = sortedPoints.size
        var f = Array(n) { DoubleArray(n) }

        for (i in 0 until n) {
            f[i][0] = sortedValues[i]
        }

        for (j in 1 until n) {
            for (i in 0 until n - j) {
                f[i][j] = (f[i + 1][j - 1] - f[i][j - 1]) / (sortedPoints[i + j] - sortedPoints[i])
            }
        }

        return DoubleArray(n) { i -> f[0][i] }
    }

    fun newtonInterpolate(): Polynome {
        var coefficients = dividedDifferences()
        var sortedPairs = points.zip(values).sortedBy { it.first }
        var sortedPoints = sortedPairs.map { it.first }
        var n = sortedPoints.size

        // Использование функции run
        var result = coefficients[0].run {
            Polynome(doubleArrayOf(this))
        }

        for (i in 1 until n) {
            // Использование функции let
            val term = coefficients[i].let { coeff ->
                var polyTerm = Polynome(doubleArrayOf(coeff))

                for (j in 0 until i) {
                    polyTerm = polyTerm * Polynome(doubleArrayOf(-sortedPoints[j], 1.0))
                }
                polyTerm
            }

            result = result + term
        }

        return result
    }

    fun newtonValue(x: Double): Double {
        return newtonInterpolate().valueAt(x)
    }
}

fun f(x: Double): Double = 3*x*x+2*x-6
fun g(x: Double): Double = sin(x)
fun h(x: Double): Double = cos(x)

fun main(){
    // Использование let
    val a = Polynome(doubleArrayOf(7.0, 1.0, 2.0, 3.7)).let {poly ->
        println("Полином: $poly")
        poly
    }

    // Использование run
    val b = a.run {
        println("Производная: ${derivative()}")
        println("Значение полинома в точке 2: ${valueAt(2.0)}")
        println()
        derivative()
    }

    val equispaced = EquispacedNodes(0.0, 2.0, 5)
    val chebyshev = ChebyshevNodes(5, 0.0, 2.0)
    val randomNodes = RandomNodes(5, 0.0, 2.0)

    // Использование let
    equispaced.nodes.let { nodes ->
        println("Равноотстоящие узлы: $nodes")
    }

    chebyshev.nodes.run {
        println("Узлы Чебышева: $this")
    }

    println("Рандомные узлы: ${randomNodes.nodes}")
    println()

    val PointsForNP = listOf(0.0, 1.0, 2.0, 3.0, 4.0)
    val ValuesForNP = PointsForNP.map { f(it) } // Лямбда-выражение
    val newton = Newton(PointsForNP, ValuesForNP)
    println("Точки интерполяции: $PointsForNP")
    println("Значения в точках: $ValuesForNP")

    // Использование let
    val interpolatingPoly = newton.newtonInterpolate().let { poly ->
        println("Интерполяционный полином: $poly")
        poly
    }

    val testPoint = 2.5
    // Использование run
    testPoint.run {
        println("Значение в точке $this: ${newton.newtonValue(this)}")
        println("Точное значение f($this): ${f(this)}")
    }
    println()

    val sinPoints = listOf(0.0, PI/4, PI/2, 3*PI/4, PI)
    val sinValues = sinPoints.map { g(it) } // Лямбда-выражение
    val sinNewton = Newton(sinPoints, sinValues)

    // Использование let
    val sinPoly = sinNewton.let { newton ->
        val poly = newton.newtonInterpolate()
        println("Интерполяция sin(x): $poly")
        poly
    }

    // Использование run
    (PI/6).run {
        println("Значение в PI/6: ${sinNewton.newtonValue(this)}")
        println("Точное значение sin(PI/6): ${sin(this)}")
    }
    println()

    val cosPoints = listOf(0.0, PI/4, PI/2, 3*PI/4, PI)
    val cosValues = cosPoints.map { h(it) }
    val cosNewton = Newton(cosPoints, cosValues)

    // Использование run
    cosNewton.run {
        val cosPoly = newtonInterpolate()
        println("Интерполяция cos(x): $cosPoly")

        (PI/6).let { x ->
            println("Значение в PI/6: ${newtonValue(x)}")
            println("Точное значение cos(PI/6): ${cos(x)}")
        }
    }
}
