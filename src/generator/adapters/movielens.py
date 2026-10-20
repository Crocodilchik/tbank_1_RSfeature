"""MovieLens 20M local datetime mirror, ratings and dated user tags."""

import polars as pl

from src.generator.adapters.base import BaseAdapter, DataValidationError, item_metadata


class MovieLensAdapter(BaseAdapter):
    """Keep explicit ratings separate from undated genome assumptions."""

    def load_raw(self) -> pl.LazyFrame:
        """Read the singular-name local ratings CSV, not the official Unix schema."""
        return self.scan("ratings", {"userId": pl.Int64, "movieId": pl.Int64, "rating": pl.Float64, "timestamp": pl.String})

    def observations_lazy(self) -> pl.LazyFrame:
        """Preserve original ratings and normalize UTC-assumed timestamps."""
        raw = self.load_raw().unique(subset=["userId", "movieId", "rating", "timestamp"], keep="first", maintain_order=True)
        frame = raw.rename({"userId": "user_id", "movieId": "item_id"}).with_columns(
            pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC").dt.epoch("s").alias("timestamp"),
            pl.lit(None, dtype=pl.Int64).alias("timestamp_ms"),
            pl.concat_str([pl.lit("rating_"), pl.col("rating").cast(pl.String)]).alias("event_type"),
            pl.when(pl.col("rating") >= 4).then(2).when(pl.col("rating") >= 3).then(1).otherwise(0).cast(pl.Int8).alias("label"))
        return self.event_identity(frame)

    def validate(self) -> list[str]:
        """Reject ratings outside the declared half-star scale."""
        errors = super().validate()
        if errors:
            return errors
        bad = self.load_raw().filter(~pl.col("rating").is_in([x / 2 for x in range(1, 11)])).limit(1).collect(engine="streaming")
        return ["MovieLens rating outside half-star 0.5..5 scale"] if bad.height else []

    def metadata_lazy(self) -> pl.LazyFrame:
        """Export static catalog fields; genome requires its own diagnostic pipeline."""
        movies = self.scan("movies", {"movieId": pl.Int64, "title": pl.String, "genres": pl.String}).rename({"movieId": "item_id"})
        links = self.scan("links", {"movieId": pl.Int64, "imdbId": pl.Int64, "tmdbId": pl.Int64}).rename({"movieId": "item_id"})
        if movies.select(pl.col("item_id").is_duplicated().any()).collect().item():
            raise DataValidationError("duplicate movieId in MovieLens catalog")
        expressions = [(movies, "title", pl.col("title")),
                       (movies, "genres", pl.col("genres").str.split("|").list.filter(pl.element() != "(no genres listed)")),
                       (movies, "release_year", pl.col("title").str.extract(r"\((\d{4})\)\s*$", 1).cast(pl.Int64, strict=False)),
                       (links, "imdb_id", pl.col("imdbId")), (links, "tmdb_id", pl.col("tmdbId"))]
        parts = [item_metadata(f, "movielens", key, expr) for f, key, expr in expressions]
        if self.settings.genome_scores_enabled:
            genome = self.scan("genome_scores", {"movieId": pl.Int64, "tagId": pl.Int64, "relevance": pl.Float64}).rename({"movieId": "item_id"})
            parts.append(item_metadata(genome, "movielens", pl.concat_str([pl.lit("genome_"), pl.col("tagId").cast(pl.String)]), pl.col("relevance")))
        return pl.concat(parts)

    def tags_lazy(self) -> pl.LazyFrame:
        """Return dated tags as a separate observation stream, never static genome."""
        return self.scan("tags", {"userId": pl.Int64, "movieId": pl.Int64, "tag": pl.String, "timestamp": pl.String}).select(
            pl.col("userId").alias("user_id"), pl.col("movieId").alias("item_id"), "tag",
            pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", time_zone="UTC").dt.epoch("s").alias("timestamp"))
