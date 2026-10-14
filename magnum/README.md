# Описание бизнес модели на языке случайных процессов

## Про события
Вот пример события
- в нем секции do, trigger. В секции do описывается как изменяются параметры. В секции trigger правило по которому события рождаются
- события могут быть параметризованы

```
event StartDay {
    do {
        number_day = number_day + 1
    }
    trigger {
        every 1d
    }
}
```

Вот еще один пример события.
- В нем событие происходит случайно, по закону пуассона, с частотой в среднем 1 раз в день
```
event VisitClient {
    do {
        count_client = count_client + 1
    }
    trigger {
        arrival poisson(lambda=1, per=1d)
    }
}
```

Вот еще один пример, событие пораждает событие
```
event child(param) {
    do {
        sum = sum + param
    }
}

event parent {
    trigger {
        every 1h
    }
    emit child(3)
}

```

Вот пример, когда событие превращается в одно из событий с некоторой вероятностью

```
event OneHourDelay {}
event TwoHourDelay {}

event RandomBranch {
    choose {
        probability_one: emit OneHourDelay after 1h
        probability_two: emit TwoHourDelay after 2h
    }
    trigger {
        every 1m
    }
}
```

## Про функции

```
# time передается в днях
function compute_hour(time) {
    day = floor(time)
    hour = floor((time - day) * 24)
    return hour
}
```

## Про секцию do
- формулы выполняются в порядке, описанном в do
- в формулах можно использовать мгновенные переменные, которые существуют только в рамках события
- есть стандартные операции +, -, *, /, ^(возведение в степень)
- в секции do можно вызывать функции

```
do {
    # мгновенные переменные описываются как instant[temp]
    instant[temp] = 1.5
    sum = sum + instant[temp]
    hour = call compute_hour(intstant[temp])
}
```

## Про переменные
переменные бывают:
- дробными значениями
- массивами

переменные объявляются в секции do или в секции controls

## Про секцию controls

в ней может быть 2 подсекции. default, limits. В default указаны дефолтные значения параметров, в limits ограничения на параметры.
- границы параметров могут включаться и не включаться
- некоторые параметры могут быть неограничены
- параметря controls могут использоваться в do
- границы для массива задаются сразу для всех параметров

```
controls {
    default {
        average_client_per_hour = 6
        probability_leave = 0.01
        example = 0.5
        mat3x3_autogen = create_array(3, 3, value=1)
        mat3x3_manual = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    }
    limits {
        probability_leave in [0, 1]
        example in (0, 1]
        mat3x3_autogen in [-10, 10]
    }
}
```

## Про массивы
в секции controls показано как можно создавать массивы.
Обращаться к ним можно так:
```
    arr[0][0]
```

## Про порядок работы секций в event
- сначала trigger
- потом do
- потом choose
- потом emit


## Про описание модели
Вот пример описания модели
- сначала парсится секция controls
- сначала парсятся все функции
- потом парсятся события

```
model "Простой пример сумматора" {

    controls {
        default {
            value = 1
        }
        limits {
            value in [-10, 100]
        }
    }

    event _init {
        do {
            summa = 0
        }
        trigger {
            on start
        }
    }

    event _set_value(new_value) {
        do {
            value = new_value
        }
    }

    event _accumulator {
        do {
            summa = summa + value
            instant[probability_is_last_event] = 0.001
        }
        trigger {
            every 1d
        }
        choose {
            instant[probability_is_last_event]: emit _set_value(0)
        }
    }

}

```

## Про результат работы модели
модель запускается с:
- заданными контролами
- заданным сидом
- заданным ограничением по времени в днях (например на 365 дней - горизонт модели)

модель должна вернуть состояния всех переменных на указанный период в днях
