# Документация Fiml на русском языке

Начните с [обзора и установки](README.md) или [руководства Python](crates/fiml-python/README.md).

Это переводы документации репозитория. В начале каждого перевода есть ссылка на
английский оригинал. Примеры кода, команды и идентификаторы API сохранены.
Даты, статусы и выводы исторических обзоров отражают состояние на момент их написания;
они не заменяют актуальные руководства.

## Обзор и справочники

- [Fiml](README.md)
- [Журнал изменений](CHANGELOG.md)
- [Словарь предметной области](CONTEXT.md)
- [fiml (привязки Python)](crates/fiml-python/README.md)
- [Глоссарий](glossary.md)
- [Схема типов и зависимостей проекта](project-schema.md)

## Конвейеры и спецификации

- [Состояние конвейера](pipeline.md)
- [Несколько выходов конвейера](multiple_output.md)
- [Несколько выходов конвейера: архитектура графа](multiple_output-arc.md)
- [Обученные преобразования sklearn в конвейере FIML](pipeline_transformer.md)
- [Средние по отсчётам — преобразования конвейера](sample-average-migration.md)
- [Привязки fiml Python — спецификация использования](specs/python.md)
- [Python API DataFrame сделок](specs/python2027-07-14.md)
- [Требования: генерация целей по временному горизонту для событийных торговых данных](time_horizon_target_requirements.md)

## Архитектурные решения

- [ADR 0001: Сделки как граница высокоуровневого DataFrame API Python](adr/0001-python-trade-dataframe-boundary.md)
- [ADR 0002: Единое определение и компиляция индикаторов](adr/0002-unify-indicator-definition-and-compilation.md)
- [ADR 0003: Семантика неполного окна SMA по отсчётам](adr/0003-define-sample-sma-partial-window-semantics.md)
- [ADR 0004: Настраиваемый прогрев оконных индикаторов](adr/0004-configurable-window-indicator-warmup.md)
- [ADR 0005: JSON-контракт спецификации вектора признаков](adr/0005-design-the-feature-extractor-spec-json-contract.md)
- [ADR 0006: Идентичность символов без учёта регистра](adr/0006-make-symbol-identity-case-insensitive.md)
- [ADR 0007: Закрытая среда исполнения экстрактора](adr/0007-close-the-extractor-runtime.md)
- [ADR 0008: Сериализация спецификации конвейера](adr/0008-serialize-the-pipeline-spec.md)
- [ADR 0009: Вычисление признаков стакана](adr/0009-derive-order-book-features.md)

## Выпуски

- [Список проверок выпуска 0.3.0](release-check-list.md)
- [Fiml 0.1.0](releases/v0.1.0.md)
- [Fiml 0.2.0](releases/v0.2.0.md)
- [Fiml 0.3.0](releases/v0.3.0.md)

## Разработка и исторические обзоры

- [Аудит публичного API Rust для v0.1.0](public-api-audit.md)
- [Состояние рефакторинга](refactor_staus.md)
- [Порядок рефакторинга и реализации — 2026-10-14](refactor-2026-10-14.md)
- [Обзор привязок Python — проблемы и улучшения](issues/2026-07-12-python.md)
- [Обзор проекта fiml](issues/2026-07-25-claude-review.md)
- [Версионированные преобразования признаков и замена конвейера](issues/2026-08-28-features.md)
- [Обзор кодовой базы — 2026-09-05](issues/2026-09-05-astra.md)
- [Общий эталон воспроизведения нескольких выходов](tests/fixtures/multiple_output_parity/README.md)

## Рекомендации агентам

- [Документация предметной области](agents/domain.md)
- [Трекер задач: GitHub](agents/issue-tracker.md)
- [Метки разбора задач](agents/triage-labels.md)
