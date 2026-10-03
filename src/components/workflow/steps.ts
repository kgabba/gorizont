import type { ComponentType, SVGProps } from "react";
import {
  IconCube,
  IconDatabase,
  IconEllipsoid,
  IconLayers,
  IconPyramid,
  IconReport,
  IconSliders,
  IconStats,
  IconTerrain,
  IconValidate,
  IconVariogram,
} from "./icons";

type Icon = ComponentType<SVGProps<SVGSVGElement>>;

export type SubStep = {
  id: string;
  num: string;
  title: string;
  Icon: Icon;
};

export type WorkflowStep = {
  id: string;
  num: string;
  title: string;
  Icon: Icon;
  panelTitle: string;
  panelLead: string;
  /** Stage that Горизонт automates */
  isProduct?: boolean;
  productBadge?: string;
  subSteps?: SubStep[];
  output?: string;
  cta?: { label: string; href: string };
};

export const WORKFLOW_STEPS: WorkflowStep[] = [
  {
    id: "data",
    num: "01",
    title: "Исходные данные бурения и опробования",
    Icon: IconDatabase,
    panelTitle: "Исходные данные бурения и опробования",
    panelLead:
      "Сбор и структурирование первичных данных скважин: координаты, интервалы опробования, содержания и служебные атрибуты для дальнейшего анализа.",
  },
  {
    id: "qc",
    num: "02",
    title: "Контроль качества и валидация",
    Icon: IconValidate,
    panelTitle: "Контроль качества и валидация",
    panelLead:
      "Проверка целостности массива, дублей, выбросов и согласованности координат — до интерпретации и статистической подготовки.",
  },
  {
    id: "domains",
    num: "03",
    title: "Геологическая интерпретация и выделение доменов",
    Icon: IconTerrain,
    panelTitle: "Геологическая интерпретация и выделение доменов",
    panelLead:
      "Формирование геологических доменов и границ, в пределах которых пространственная структура минерализации считается однородной.",
  },
  {
    id: "composite",
    num: "04",
    title: "Композитирование и статистическая подготовка",
    Icon: IconStats,
    panelTitle: "Композитирование и статистическая подготовка",
    panelLead:
      "Приведение проб к единой длине композита, описательная статистика и подготовка распределения для вариографии и оценки.",
  },
  {
    id: "spatial",
    num: "05",
    title: "Подготовка параметров пространственной оценки",
    Icon: IconLayers,
    isProduct: true,
    productBadge: "Целевой этап нашей платформы",
    panelTitle: "Автоматизация подготовки параметров пространственной оценки",
    panelLead:
      "Именно этот этап закрывает наш софт: от вариографии до рекомендуемых параметров кригинга и отчёта по качеству модели.",
    subSteps: [
      {
        id: "variography",
        num: "1",
        title: "Вариографический анализ",
        Icon: IconVariogram,
      },
      {
        id: "anisotropy",
        num: "2",
        title: "Анизотропия и эллипсоид поиска",
        Icon: IconEllipsoid,
      },
      {
        id: "tuning",
        num: "3",
        title: "Автоподбор параметров IDW / Ordinary Kriging",
        Icon: IconSliders,
      },
      {
        id: "validation",
        num: "4",
        title: "Пространственная валидация и отчёт",
        Icon: IconReport,
      },
    ],
    output:
      "рекомендуемые параметры оценки для ГГИС и отчёт по качеству модели",
    cta: { label: "Начать анализ", href: "/#analyze" },
  },
  {
    id: "block",
    num: "06",
    title: "Оценка блочной модели",
    Icon: IconCube,
    panelTitle: "Оценка блочной модели",
    panelLead:
      "Интерполяция содержаний в блочную модель с использованием подобранных параметров поиска и вариограммы.",
  },
  {
    id: "resources",
    num: "07",
    title: "Оценка ресурсов",
    Icon: IconPyramid,
    panelTitle: "Оценка ресурсов",
    panelLead:
      "Классификация и подсчёт ресурсов на основе блочной модели, доменов и принятых критериев отчётности.",
  },
];
