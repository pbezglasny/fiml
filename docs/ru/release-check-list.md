[English](../release-check-list.md)

<a id="release-020-checklist"></a>

# Список проверок выпуска 0.2.0

Целевой тег: `v0.2.0`. Пакеты Rust и Python выпускаются вместе.
Предыдущий [v0.1.0](https://github.com/pbezglasny/fiml/releases/tag/v0.1.0)
опубликован 2026-09-15 из `57619d5`.

<a id="release-records"></a>

## Записи выпуска

- [x] Установить версии обоих Cargo-пакетов `0.2.0`; Python берёт версию из манифеста привязок.
- [x] Описать возможности и несовместимости в [CHANGELOG.md](CHANGELOG.md).
- [x] Подготовить [заметки GitHub](releases/v0.2.0.md), читаемые `release.yml`.
- [x] Обновить установку и ссылку на [миграцию](sample-average-migration.md).
- [x] Сохранить Rust 1.89, CPython 3.12+ и пять целей wheel.
- [ ] Заменить `Unreleased` для 0.2.0 датой публикации в журнале.

<a id="local-validation"></a>

## Локальная проверка

Проверено 2026-10-02 в рабочем дереве выпуска. Повторить на конечном чистом коммите
до тега. Локальный `--allow-dirty` нужен только для проверки подготовленных
незакоммиченных изменений; CI упаковывает чистую копию.

- [x] `cargo fmt --all -- --check`
- [x] `cargo +1.89.0 test --workspace --all-features`: 278 тестов и 2 doctest.
- [x] `cargo +1.89.0 clippy --workspace --all-targets --all-features -- -D warnings`
- [x] `RUSTDOCFLAGS="-D warnings -D missing_docs" cargo +1.89.0 doc -p fiml --all-features --no-deps`
- [x] `make test-python`: 489 тестов на CPython 3.13.
- [x] `make test-notebook`: исполнение и экспорт marimo.
- [x] `cargo +1.89.0 package -p fiml --all-features --allow-dirty`: 88 файлов;
  последний локальный запуск с `--offline` после получения зависимостей.
- [x] Собрать release wheel и исходный дистрибутив Maturin 1.14.1.
- [x] Проверить версии, исходники, README и лицензию в crate, wheel, исходном архиве.
- [x] Установить wheel в новое CPython 3.13; 489 тестов и quickstart проходят.
- [x] Установить исходный дистрибутив в отдельное новое окружение и запустить quickstart.
- [x] Собрать/запустить Rust-пример README с упакованным crate и экспортом Python;
  оба возвращают `1.224745`.

Локальный Linux x86-64 wheel проверяет только эту машину. Процесс выпуска собирает
manylinux 2.17 x86-64/AArch64, macOS x86-64/AArch64, Windows x86-64;
проверки этих платформ остаются обязательной частью CI.

<a id="final-ci-and-publishing"></a>

## Финальный CI и публикация

Отправка `v0.2.0` запускает `.github/workflows/release.yml`, публикующий crates.io,
PyPI и выпуск GitHub. Ручной запуск workflow собирает без публикации.
Не загружайте пакеты вручную заранее.

- [ ] Закоммитить изменения и получить успешный CI именно для этого коммита.
- [ ] Вручную запустить workflow на нём; проверить crate, пять wheel, исходный
  дистрибутив, включая API-тесты Linux wheel.
- [ ] Убедиться в готовности `CARGO_REGISTRY_TOKEN` и PyPI Trusted Publishing.
- [ ] Поставить `v0.2.0` на проверенный коммит и отправить тег.
- [ ] Проверить `fiml` 0.2.0 в обоих реестрах и семь артефактов с нужными заметками GitHub.
- [ ] Проверить новые установки `cargo add fiml@0.2.0`, `pip install fiml==0.2.0`,
  документацию и метаданные реестров.
