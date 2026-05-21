use super::{ModelError, DEFAULT_MAX_PREFIX_BYTES};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum MetadataValue {
    String(String),
    Bool(bool),
    Number,
    Array,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum GgufValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GgufValueType {
    #[cfg(test)]
    pub(super) fn as_u32(self) -> u32 {
        match self {
            Self::Uint8 => 0,
            Self::Int8 => 1,
            Self::Uint16 => 2,
            Self::Int16 => 3,
            Self::Uint32 => 4,
            Self::Int32 => 5,
            Self::Float32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::Uint64 => 10,
            Self::Int64 => 11,
            Self::Float64 => 12,
        }
    }

    pub(super) fn from_u32(value: u32) -> Result<Self, ModelError> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(ModelError::InvalidGgufMetadata(format!(
                "unsupported value type {value}"
            ))),
        }
    }

    fn scalar_size(self) -> Option<usize> {
        match self {
            Self::Uint8 | Self::Int8 | Self::Bool => Some(1),
            Self::Uint16 | Self::Int16 => Some(2),
            Self::Uint32 | Self::Int32 | Self::Float32 => Some(4),
            Self::Uint64 | Self::Int64 | Self::Float64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

pub(super) fn read_value(
    bytes: &[u8],
    offset: &mut usize,
    value_type: GgufValueType,
) -> Result<MetadataValue, ModelError> {
    match value_type {
        GgufValueType::String => Ok(MetadataValue::String(read_string(bytes, offset)?)),
        GgufValueType::Bool => {
            let end = checked_end(*offset, 1)?;
            require(bytes, end)?;
            let value = bytes[*offset] != 0;
            *offset = end;
            Ok(MetadataValue::Bool(value))
        }
        GgufValueType::Array => {
            skip_array(bytes, offset)?;
            Ok(MetadataValue::Array)
        }
        _ => {
            skip_value(bytes, offset, value_type)?;
            Ok(MetadataValue::Number)
        }
    }
}

pub(super) fn skip_value(
    bytes: &[u8],
    offset: &mut usize,
    value_type: GgufValueType,
) -> Result<(), ModelError> {
    match value_type {
        GgufValueType::String => {
            let _ = read_string(bytes, offset)?;
        }
        GgufValueType::Array => skip_array(bytes, offset)?,
        _ => {
            let size = value_type.scalar_size().ok_or_else(|| {
                ModelError::InvalidGgufMetadata("unsupported scalar type".to_string())
            })?;
            advance_offset_with_require(bytes, offset, size)?;
        }
    }
    Ok(())
}

fn skip_array(bytes: &[u8], offset: &mut usize) -> Result<(), ModelError> {
    let value_type = GgufValueType::from_u32(read_u32(bytes, *offset)?)?;
    advance_offset(offset, 4)?;
    let len = read_u64_as_usize(bytes, *offset)?;
    advance_offset(offset, 8)?;
    if value_type == GgufValueType::String {
        for _ in 0..len {
            let _ = read_string(bytes, offset)?;
        }
        return Ok(());
    }
    let Some(item_size) = value_type.scalar_size() else {
        return Err(ModelError::InvalidGgufMetadata(
            "nested GGUF arrays are not supported".to_string(),
        ));
    };
    let byte_len = len
        .checked_mul(item_size)
        .ok_or_else(|| ModelError::InvalidGgufMetadata("array length overflow".to_string()))?;
    advance_offset_with_require(bytes, offset, byte_len)?;
    Ok(())
}

pub(super) fn read_string(bytes: &[u8], offset: &mut usize) -> Result<String, ModelError> {
    let len = read_u64_as_usize(bytes, *offset)?;
    advance_offset(offset, 8)?;
    let end = checked_end(*offset, len)?;
    require(bytes, end)?;
    let value = std::str::from_utf8(&bytes[*offset..end])
        .map_err(|_| ModelError::InvalidGgufMetadata("string is not UTF-8".to_string()))?
        .to_string();
    *offset = end;
    Ok(value)
}

pub(super) fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ModelError> {
    let end = checked_end(offset, 4)?;
    require(bytes, end)?;
    let mut value = [0_u8; 4];
    value.copy_from_slice(&bytes[offset..end]);
    Ok(u32::from_le_bytes(value))
}

pub(super) fn read_u64_as_usize(bytes: &[u8], offset: usize) -> Result<usize, ModelError> {
    let end = checked_end(offset, 8)?;
    require(bytes, end)?;
    let mut value = [0_u8; 8];
    value.copy_from_slice(&bytes[offset..end]);
    let value = u64::from_le_bytes(value);
    usize::try_from(value)
        .map_err(|_| ModelError::InvalidGgufMetadata("length does not fit usize".to_string()))
}

fn checked_end(offset: usize, len: usize) -> Result<usize, ModelError> {
    offset
        .checked_add(len)
        .ok_or_else(|| ModelError::InvalidGgufMetadata("metadata offset overflow".to_string()))
}

pub(super) fn advance_offset(offset: &mut usize, len: usize) -> Result<(), ModelError> {
    *offset = checked_end(*offset, len)?;
    Ok(())
}

pub(super) fn advance_offset_with_require(
    bytes: &[u8],
    offset: &mut usize,
    len: usize,
) -> Result<(), ModelError> {
    let end = checked_end(*offset, len)?;
    require(bytes, end)?;
    *offset = end;
    Ok(())
}

fn require(bytes: &[u8], end: usize) -> Result<(), ModelError> {
    if end <= bytes.len() {
        return Ok(());
    }
    if bytes.len() >= DEFAULT_MAX_PREFIX_BYTES {
        return Err(ModelError::GgufMetadataTooLarge {
            max_bytes: DEFAULT_MAX_PREFIX_BYTES,
        });
    }
    Err(ModelError::InvalidGgufMetadata(
        "metadata is truncated".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_helpers_report_overflow_and_truncation() {
        let mut offset = usize::MAX;
        let overflow = advance_offset(&mut offset, 1).expect_err("offset overflow");
        assert!(
            matches!(overflow, ModelError::InvalidGgufMetadata(message) if message.contains("offset overflow"))
        );

        let mut offset = 2;
        let truncated =
            advance_offset_with_require(&[0_u8; 2], &mut offset, 1).expect_err("truncated");
        assert!(
            matches!(truncated, ModelError::InvalidGgufMetadata(message) if message.contains("truncated"))
        );
    }
}
